"""
MalaStar optimizer - reimplemented under the new optimization framework.

MALA* (Metropolis-Adjusted Langevin Algorithm Star) from GraspQP paper.
This is a gradient-based MCMC sampler with:
- RMSProp-style gradient normalization
- Temperature annealing with z_score adjustment
- Metropolis-Hastings accept/reject
- Optional contact point switching (for grasp generation mode)

The key insight is that gradients must flow through hand_model properties.
We compute energy using the costs, but the gradient accumulates on
hand_model.hand_pose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import Normal

from .base import Optimizer

# Standard normal distribution for z_score -> probability conversion
_normal = Normal(0, 1)

if TYPE_CHECKING:
    from ..problem import OptimizationProblem
    from ..state import TrajectoryState


class MalaStarOptimizer(Optimizer):
    """
    MALA* optimizer from GraspQP, adapted for the new framework.

    This optimizer:
    1. Uses gradients from problem.total_energy() for proposals
    2. Applies Metropolis-Hastings accept/reject
    3. Optionally samples new contact points
    """

    def __init__(
        self,
        fix_contacts: bool = False,
        switch_possibility: float = 0.4,
        starting_temperature: float = 18.0,
        temperature_decay: float = 0.95,
        annealing_period: int = 30,
        step_size: float = 0.005,
        stepsize_period: int = 50,
        mu: float = 0.98,
        clip_grad: bool = False,
        profiler=None,  # Optional profiler for timing
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.fix_contacts = fix_contacts
        self.switch_possibility = switch_possibility
        self.starting_temperature = starting_temperature
        self.temperature_decay = temperature_decay
        self.annealing_period = annealing_period
        self.step_size = step_size
        self.stepsize_period = stepsize_period
        self.mu = mu
        self.clip_grad = clip_grad
        self.profiler = profiler

        # State variables (initialized on first step)
        self._ema_grad: Optional[Tensor] = None
        self._per_batch_step: Optional[Tensor] = None
        self._current_energy: Optional[Tensor] = None
        self._initialized: bool = False
        self._debug: bool = False  # Set to True for verbose output

        # Cached objects to avoid repeated allocation (set on first use)
        self._cached_object_states: Optional[Tensor] = None
        self._cached_state: Optional[Any] = None  # TrajectoryState

    def _profile_section(self, name: str):
        """Context manager for profiling a section."""
        from contextlib import nullcontext

        if self.profiler is not None:
            return self.profiler.section(name)
        return nullcontext()

    def step(
        self,
        state: "TrajectoryState",
        problem: "OptimizationProblem",
        z_score: Optional[Tensor] = None,
    ) -> "TrajectoryState":
        """
        Perform one MALA* optimization step.

        Args:
            state: Current trajectory state
            problem: Optimization problem with costs
            z_score: Optional per-batch z-score for adaptive temperature.
                     Higher z_score (worse outlier) -> higher temperature -> more exploration.
                     Shape: (B,). Computed as (energy - mean) / std per object batch.

        Flow:
        1. If first call: compute initial energy and gradient, return
        2. Otherwise: use gradient to propose, compute new energy, accept/reject
        """
        device = state.device
        hand_model = problem.context.hand_model
        B = state.B

        # Initialize per-batch step counter on first call
        if self._per_batch_step is None:
            self._per_batch_step = torch.zeros(B, dtype=torch.long, device=device)

        # Initialize EMA on first call
        D_hand = hand_model.hand_pose.shape[1]
        if self._ema_grad is None:
            self._ema_grad = torch.zeros(D_hand, dtype=torch.float, device=device)

        # =====================================================================
        # First call: compute initial energy and gradient (like fit.py's pre-loop)
        # fit.py does: backward() then zero_grad() before the loop
        # Then the first loop iteration uses zero grad for proposal
        # =====================================================================
        if not self._initialized:
            self._compute_energy_and_grad(problem, hand_model)
            if self._debug:
                print(f"\n[MalaStar Init] energy={self._current_energy.mean().item():.2f}")
                if hand_model.hand_pose.grad is not None:
                    print(f"  grad norm={hand_model.hand_pose.grad.norm().item():.4f}")
            # Zero gradient like fit.py's zero_grad() before the loop
            if hand_model.hand_pose.grad is not None:
                hand_model.hand_pose.grad.zero_()
            self._initialized = True
            # DON'T return early - continue to do the full step with zero gradient
            # This matches fit.py where first iteration still resamples contacts

        # =====================================================================
        # 1. Try step: propose new parameters using gradient
        # =====================================================================
        with self._profile_section("try_step"):
            # Compute step size with decay (use current step BEFORE incrementing - matches fit.py)
            s = self.step_size * (self.temperature_decay ** (self._per_batch_step // self.stepsize_period))
            step_size = s.unsqueeze(-1)  # (B, 1)

            # Get gradient (save for potential restore on reject)
            grad = hand_model.hand_pose.grad
            if grad is None:
                grad = torch.zeros_like(hand_model.hand_pose)
            old_grad = grad.clone()  # Save for restoration on reject

            # Clip gradients if configured
            if self.clip_grad:
                grad = grad.clamp(-100, 100)
                grad[torch.isnan(grad)] = 0

            # Update EMA of squared gradients (RMSProp-style)
            mean_grad_sq = (grad**2).mean(0)  # (D_hand,)
            self._ema_grad = self.mu * mean_grad_sq + (1 - self.mu) * self._ema_grad

            # Handle NaN in EMA
            if self._ema_grad.isnan().any():
                self._ema_grad[torch.isnan(self._ema_grad)] = 0

            # Compute normalized gradient step
            normalized_grad = grad / (torch.sqrt(self._ema_grad) + 1e-6)

            # Propose new hand pose
            # CRITICAL: Do NOT use .detach() on hand_pose! This preserves requires_grad
            # so that set_parameters clones a tensor with requires_grad=True, enabling
            # gradient flow through FK and contact computation.
            # normalized_grad comes from .grad which doesn't have requires_grad, so
            # subtracting it doesn't create unwanted gradient paths.
            proposed_pose = hand_model.hand_pose - step_size * normalized_grad

            # Handle NaN in proposal (in no_grad to avoid gradient issues)
            with torch.no_grad():
                if proposed_pose.isnan().any():
                    nan_mask = proposed_pose.isnan().any(dim=-1)
                    proposed_pose[nan_mask] = hand_model.hand_pose[nan_mask]

            # =====================================================================
            # 2. Sample new contact indices (if not fixed)
            # =====================================================================
            if self.fix_contacts:
                new_contact_indices = hand_model.contact_point_indices.clone()
            else:
                new_contact_indices = self._sample_contacts(hand_model, problem.context, device)

            # =====================================================================
            # 3. Save old state (pose, contacts, contact_points, energy, grad)
            # =====================================================================
            old_hand_pose = hand_model.hand_pose.detach().clone()
            old_contact_indices = hand_model.contact_point_indices.clone()
            old_contact_points = hand_model.contact_points.clone()
            old_global_translation = hand_model.global_translation.clone()
            old_global_rotation = hand_model.global_rotation.clone()
            old_energy = self._current_energy.clone()

            # =====================================================================
            # 4. Set new parameters
            # =====================================================================
            # Use set_parameters which creates a fresh tensor (like fit.py)
            # Since proposed_pose has requires_grad=True, the clone in set_parameters
            # also has requires_grad=True, so FK/contact computation builds gradient graph
            hand_model.set_parameters(proposed_pose, new_contact_indices)

            # Increment step counter AFTER set_parameters (matches fit.py try_step)
            self._per_batch_step += 1

        # Zero grad before computing new energy (like fit.py's zero_grad before calculate_energy)
        if hand_model.hand_pose.grad is not None:
            hand_model.hand_pose.grad.zero_()

        # Compute new energy and gradient (creates fresh graph)
        with self._profile_section("energy"):
            new_energy = self._compute_energy_and_grad(problem, hand_model)

        # =====================================================================
        # 5. Accept/reject using Metropolis-Hastings
        # =====================================================================
        with self._profile_section("accept_step"):
            temperature = self.starting_temperature * (
                self.temperature_decay ** (self._per_batch_step // self.annealing_period)
            )

            # z_score-based temperature adjustment (like fit.py)
            # Higher z_score (worse energy relative to batch) -> higher temperature -> more exploration
            if z_score is not None:
                proba = _normal.cdf(z_score.detach())  # Convert z_score to probability [0, 1]
                temperature = temperature * (1 + proba)  # Scale temperature by (1 + proba)

            with torch.no_grad():
                alpha = torch.rand(B, dtype=torch.float, device=device)
                accept = alpha < torch.exp((old_energy - new_energy) / temperature)
                reject = ~accept

                if self._debug and self._step_count < 5:
                    print(f"\n[MalaStar Step {self._step_count}]")
                    print(f"  old_energy={old_energy.mean().item():.2f}, new_energy={new_energy.mean().item():.2f}")
                    print(
                        f"  temperature={temperature.mean().item():.4f}, accept_rate={accept.float().mean().item():.2f}"
                    )
                    print(f"  step_size={s.mean().item():.6f}, EMA={self._ema_grad.mean().item():.6f}")
                    print(f"  grad norm (before)={old_grad.norm().item():.4f}")
                    if hand_model.hand_pose.grad is not None:
                        new_grad = hand_model.hand_pose.grad
                        print(f"  grad norm (after)={new_grad.norm().item():.4f}")
                        print(
                            f"  grad trans={new_grad[:,:3].norm().item():.2f}, rot={new_grad[:,3:9].norm().item():.2f}, joints={new_grad[:,9:].norm().item():.2f}"
                        )
                    print(
                        f"  hand_pose change={((hand_model.hand_pose - old_hand_pose)**2).sum(-1).sqrt().mean().item():.6f}"
                    )

                # Restore rejected states (pose, contacts, contact_points, global transforms, energy, gradient)
                # CRITICAL: Also restore gradient for rejected samples, matching fit.py behavior
                if reject.any():
                    hand_model.hand_pose[reject] = old_hand_pose[reject]
                    hand_model.contact_point_indices[reject] = old_contact_indices[reject]
                    hand_model.contact_points[reject] = old_contact_points[reject]
                    hand_model.global_translation[reject] = old_global_translation[reject]
                    hand_model.global_rotation[reject] = old_global_rotation[reject]
                    self._current_energy[reject] = old_energy[reject]
                    # Restore old gradient for rejected samples (key for proper convergence!)
                    if hand_model.hand_pose.grad is not None:
                        hand_model.hand_pose.grad[reject] = old_grad[reject]

                # Recompute FK for consistency
                hand_model.current_status = hand_model.fk(hand_model.hand_pose[:, 9:])

        # Step counter was already incremented at the start of this function
        self._step_count += 1

        # =====================================================================
        # 6. Return updated state (sync from hand_model)
        # =====================================================================
        final_state = state.clone()
        final_state.hand_states = hand_model.hand_pose.detach().unsqueeze(1)  # (B, 1, D_hand)

        return final_state

    def _compute_energy_and_grad(self, problem: "OptimizationProblem", hand_model) -> Tensor:
        """
        Compute energy and backprop to get gradients.

        Like fit.py: hand_model is already configured via set_parameters.
        We just need to compute energy and call backward().

        Note: This method is wrapped in a profiler section ("energy") by the caller.
        We add sub-sections for more detail.
        """
        # Clear step cache
        problem.context.clear_step_cache()

        # Skip set_parameters in costs - hand is already configured
        problem.context._skip_set_parameters = True

        try:
            # Ensure hand_model.hand_pose has requires_grad
            hand_model.hand_pose.requires_grad_(True)

            B = hand_model.hand_pose.shape[0]
            device = hand_model.hand_pose.device

            # Cache object_states (identity quaternion at origin - never changes)
            if self._cached_object_states is None or self._cached_object_states.shape[0] != B:
                self._cached_object_states = torch.zeros(B, 1, 7, device=device)
                self._cached_object_states[:, :, 6] = 1.0  # Identity quaternion (w=1)

            # Get hand_states as view of current hand_pose
            hand_states = hand_model.hand_pose.unsqueeze(1)  # (B, 1, D)

            # Reuse or create TrajectoryState
            if self._cached_state is None or self._cached_state.B != B:
                from ..state import TrajectoryState

                self._cached_state = TrajectoryState(
                    hand_states=hand_states,
                    object_states=self._cached_object_states,
                )
            else:
                # Update hand_states reference (object_states stays the same)
                self._cached_state.hand_states = hand_states

            # Compute energy using costs (they access hand_model properties)
            with self._profile_section("costs"):
                energy = problem.total_energy(self._cached_state)  # (B,)

            # Store for accept/reject
            self._current_energy = energy.detach().clone()

            # Backward to get gradients on hand_model.hand_pose
            with self._profile_section("backward"):
                energy.sum().backward()

            return energy.detach()
        finally:
            # Restore normal mode
            problem.context._skip_set_parameters = False

    def _sample_contacts(
        self,
        hand_model,
        context,
        device: torch.device,
    ) -> Tensor:
        """Sample new contact indices with switch_possibility."""
        current_contacts = hand_model.contact_point_indices.clone()
        batch_size, n_contact = current_contacts.shape

        # Determine which contacts to switch
        switch_mask = torch.rand(batch_size, n_contact, device=device) < self.switch_possibility

        if context.contact_sampler is not None:
            # Use contact sampler that respects finger constraints
            batch_switch_mask = switch_mask.any(dim=1)
            if batch_switch_mask.any():
                n_switch = batch_switch_mask.sum().item()
                new_samples = context.contact_sampler.sample(n_switch, n_contact)
                switch_indices = batch_switch_mask.nonzero(as_tuple=True)[0]
                for i, idx in enumerate(switch_indices):
                    item_switch = switch_mask[idx]
                    current_contacts[idx, item_switch] = new_samples[i, item_switch]
        else:
            # Fallback: uniform sampling
            n_candidates = hand_model.n_contact_candidates
            new_indices = torch.randint(n_candidates, size=(batch_size, n_contact), device=device)
            current_contacts[switch_mask] = new_indices[switch_mask]

        return current_contacts

    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self._ema_grad = None
        self._per_batch_step = None
        self._current_energy = None
        self._initialized = False
        self._cached_object_states = None
        self._cached_state = None

    def reset_envs(self, mask: Tensor):
        """Reset specific environments."""
        if self._per_batch_step is not None:
            self._per_batch_step[mask] = 0

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        diag = super().get_diagnostics()
        diag.update(
            {
                "fix_contacts": self.fix_contacts,
                "switch_possibility": self.switch_possibility,
                "starting_temperature": self.starting_temperature,
                "temperature_decay": self.temperature_decay,
                "step_size": self.step_size,
            }
        )
        if self._per_batch_step is not None:
            diag["mean_step"] = self._per_batch_step.float().mean().item()
        return diag
