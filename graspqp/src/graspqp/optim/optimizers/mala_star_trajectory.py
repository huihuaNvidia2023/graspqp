"""
MalaStar optimizer for trajectory optimization (T > 1).

Extends the single-frame MALA* to handle multi-frame trajectories.
Key differences from mala_star.py:
- Optimizes hand_states with shape (B, T, D_hand)
- Computes energy summed over all T frames
- Accept/reject operates on entire trajectories
- Contact indices are FIXED for all frames (shared per trajectory)

This optimizer treats the trajectory as a single optimization unit.
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


class MalaStarTrajectoryOptimizer(Optimizer):
    """
    MALA* optimizer adapted for trajectory optimization (T > 1).

    Optimizes entire trajectories as units. Each trajectory consists of
    T frames, and the optimizer proposes changes to ALL frames simultaneously.

    Key design decisions:
    - Contact indices are FIXED for all frames in a trajectory
    - Accept/reject applies to the entire trajectory (all T frames together)
    - Energy is summed over frames, gradients propagate through all frames
    """

    def __init__(
        self,
        fix_contacts: bool = True,  # Usually True for trajectory refinement
        switch_possibility: float = 0.0,  # Usually 0 for trajectory mode
        starting_temperature: float = 18.0,
        temperature_decay: float = 0.95,
        annealing_period: int = 30,
        step_size: float = 0.005,
        stepsize_period: int = 50,
        mu: float = 0.98,
        clip_grad: bool = False,
        batch_size_per_object: Optional[int] = None,
        profiler=None,
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
        self.batch_size_per_object = batch_size_per_object
        self.profiler = profiler

        # State variables (initialized on first step)
        self._ema_grad: Optional[Tensor] = None  # Shape: (T, D_hand) or (D_hand,)
        self._per_batch_step: Optional[Tensor] = None
        self._current_energy: Optional[Tensor] = None  # (B,)
        self._initialized: bool = False
        self._debug: bool = False

        # Cached states
        self._T: Optional[int] = None
        self._D_hand: Optional[int] = None

    def _profile_section(self, name: str):
        """Context manager for profiling a section."""
        from contextlib import nullcontext

        if self.profiler is not None:
            return self.profiler.section(name)
        return nullcontext()

    def _compute_z_score(self, energy: Tensor) -> Optional[Tensor]:
        """
        Compute z-score for adaptive temperature adjustment.
        """
        if self.batch_size_per_object is None:
            return None

        B = energy.shape[0]
        if B % self.batch_size_per_object != 0:
            return None

        energy_grouped = energy.view(-1, self.batch_size_per_object)
        mean = energy_grouped.mean(dim=-1, keepdim=True)
        std = energy_grouped.std(dim=-1, keepdim=True).clamp(min=1e-6)
        z_score = ((energy_grouped - mean) / std).view(-1)
        return z_score

    def step(
        self,
        state: "TrajectoryState",
        problem: "OptimizationProblem",
    ) -> "TrajectoryState":
        """
        Perform one MALA* optimization step for trajectory.

        Args:
            state: Current trajectory state (B, T, D_hand)
            problem: Optimization problem with costs

        Returns:
            Updated trajectory state
        """
        device = state.device
        hand_model = problem.context.hand_model
        B, T, D_hand = state.hand_states.shape

        # Store dimensions for EMA
        if self._T is None:
            self._T = T
            self._D_hand = D_hand

        # Initialize per-batch step counter on first call
        if self._per_batch_step is None:
            self._per_batch_step = torch.zeros(B, dtype=torch.long, device=device)

        # Initialize EMA on first call
        if self._ema_grad is None:
            # Use per-dimension EMA across all frames
            self._ema_grad = torch.zeros(T * D_hand, dtype=torch.float, device=device)

        # =====================================================================
        # Initialization step: compute initial energy and gradients
        # =====================================================================
        if not self._initialized:
            with self._profile_section("init"):
                # Compute initial energy (sum over all frames)
                energy, grad = self._compute_energy_and_grad(state, problem)
                self._current_energy = energy  # (B,)

                if self._debug:
                    print(f"\n[MalaStarTrajectory Init] energy={energy.mean().item():.2f}")
                    if grad is not None:
                        print(f"  grad norm={grad.norm().item():.4f}")

            self._initialized = True

        # =====================================================================
        # 1. Propose new trajectory
        # =====================================================================
        with self._profile_section("try_step"):
            # Step size with decay
            s = self.step_size * (self.temperature_decay ** (self._per_batch_step // self.stepsize_period))
            step_size = s.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1) for (B, T, D)

            # Get gradient for current state
            _, grad = self._compute_energy_and_grad(state, problem)  # grad: (B, T, D)

            if grad is None:
                grad = torch.zeros_like(state.hand_states)

            old_grad = grad.clone()

            # Clip gradients if configured
            if self.clip_grad:
                grad = grad.clamp(-100, 100)
                grad[torch.isnan(grad)] = 0

            # Update EMA of squared gradients
            grad_flat = grad.reshape(B, -1)  # (B, T*D)
            mean_grad_sq = (grad_flat**2).mean(0)  # (T*D,)
            self._ema_grad = self.mu * mean_grad_sq + (1 - self.mu) * self._ema_grad

            # Handle NaN in EMA
            if self._ema_grad.isnan().any():
                self._ema_grad[torch.isnan(self._ema_grad)] = 0

            # Normalized gradient step
            ema_expanded = self._ema_grad.view(1, T, D_hand)  # (1, T, D)
            normalized_grad = grad / (torch.sqrt(ema_expanded) + 1e-6)

            # Propose new trajectory
            proposed_hand = state.hand_states - step_size * normalized_grad  # (B, T, D)

            # Handle NaN
            with torch.no_grad():
                if proposed_hand.isnan().any():
                    nan_mask = proposed_hand.isnan().any(dim=-1).any(dim=-1)  # (B,)
                    proposed_hand[nan_mask] = state.hand_states[nan_mask]

            # =====================================================================
            # 2. Sample new contact indices (if not fixed)
            # =====================================================================
            # Contact indices are stored as (B*T, n_contacts) in hand_model but
            # are the SAME for all T frames in each trajectory.
            # For accept/reject, we work with per-trajectory indices (B, n_contacts)
            n_contacts = hand_model.contact_point_indices.shape[-1]

            # Get per-trajectory contact indices by taking first frame of each
            current_contacts_per_traj = hand_model.contact_point_indices.reshape(B, T, n_contacts)[
                :, 0
            ]  # (B, n_contacts)

            if self.fix_contacts:
                new_contacts_per_traj = current_contacts_per_traj.clone()
            else:
                new_contacts_per_traj = self._sample_contacts(
                    hand_model, problem.context, current_contacts_per_traj, device
                )

            # =====================================================================
            # 3. Save old state
            # =====================================================================
            old_hand_states = state.hand_states.detach().clone()
            old_contacts_per_traj = current_contacts_per_traj.clone()  # (B, n_contacts)
            old_energy = self._current_energy.clone()

            # Increment step counter
            self._per_batch_step += 1

        # =====================================================================
        # 4. Create proposed state and compute energy
        # =====================================================================
        with self._profile_section("energy"):
            proposed_state = state.clone()
            proposed_state.hand_states = proposed_hand

            # Expand per-trajectory contacts to (B*T, n_contacts) for FK
            new_contacts_expanded = (
                new_contacts_per_traj.unsqueeze(1).expand(B, T, n_contacts).reshape(B * T, n_contacts)
            )

            # Update contact indices in context for FK
            problem.context.set_contact_indices(new_contacts_expanded)

            new_energy, new_grad = self._compute_energy_and_grad(proposed_state, problem)

        # =====================================================================
        # 5. Accept/reject entire trajectory
        # =====================================================================
        with self._profile_section("accept_step"):
            temperature = self.starting_temperature * (
                self.temperature_decay ** (self._per_batch_step // self.annealing_period)
            )

            # z_score-based temperature adjustment
            z_score = self._compute_z_score(old_energy)
            if z_score is not None:
                proba = _normal.cdf(z_score.detach())
                temperature = temperature * (1 + proba)

            with torch.no_grad():
                alpha = torch.rand(B, dtype=torch.float, device=device)
                accept = alpha < torch.exp((old_energy - new_energy) / temperature)
                reject = ~accept

                if self._debug and self._step_count < 5:
                    print(f"\n[MalaStarTrajectory Step {self._step_count}]")
                    print(f"  old_energy={old_energy.mean():.2f}, new_energy={new_energy.mean():.2f}")
                    print(f"  accept_rate={accept.float().mean():.2f}")

                # Restore rejected trajectories (using per-trajectory contacts)
                if reject.any():
                    proposed_hand[reject] = old_hand_states[reject]
                    new_contacts_per_traj[reject] = old_contacts_per_traj[reject]  # (B, n_contacts)
                    self._current_energy[reject] = old_energy[reject]
                else:
                    self._current_energy = new_energy

                # Update accepted energies
                self._current_energy[accept] = new_energy[accept]

                # Re-expand contacts after accept/reject
                new_contacts_expanded = (
                    new_contacts_per_traj.unsqueeze(1).expand(B, T, n_contacts).reshape(B * T, n_contacts)
                )

        self._step_count += 1

        # =====================================================================
        # 6. Return updated state
        # =====================================================================
        final_state = state.clone()
        final_state.hand_states = proposed_hand.detach()

        # Update contact indices in context (expanded version for FK)
        problem.context.set_contact_indices(new_contacts_expanded)

        return final_state

    def _compute_energy_and_grad(
        self,
        state: "TrajectoryState",
        problem: "OptimizationProblem",
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        Compute energy and gradient for trajectory state.

        For trajectory optimization, we:
        1. Flatten (B, T, D) to (B*T, D) for FK
        2. Compute per-frame costs
        3. Sum costs across frames to get per-trajectory energy
        4. Backprop to get gradient on (B, T, D)

        Returns:
            Tuple of (energy (B,), gradient (B, T, D) or None)
        """
        B, T, D = state.hand_states.shape
        device = state.device
        hand_model = problem.context.hand_model

        # Clear step cache
        problem.context.clear_step_cache()

        # Make hand_states require grad for backward
        hand_states = state.hand_states.clone()
        hand_states.requires_grad_(True)

        # Flatten for FK: (B*T, D)
        flat_hand = hand_states.reshape(B * T, D)

        # Configure hand model with flattened poses
        # Contact indices should be expanded for all frames
        contact_indices = hand_model.contact_point_indices
        if contact_indices is not None and contact_indices.shape[0] == B:
            # Expand contact indices: (B, n_c) -> (B*T, n_c)
            contact_indices = contact_indices.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)

        problem.context._current_contact_indices = contact_indices

        # Skip set_parameters in costs - we'll do it here
        problem.context._skip_set_parameters = True

        try:
            # Configure FK
            with self._profile_section("fk"):
                from graspqp.utils.transforms import robust_compute_rotation_matrix_from_ortho6d

                hand_model.global_translation = flat_hand[:, :3]
                hand_model.global_rotation = robust_compute_rotation_matrix_from_ortho6d(flat_hand[:, 3:9])
                hand_model.current_status = hand_model.fk(flat_hand[:, 9:])

                # Also update hand_pose for costs that read it
                hand_model.hand_pose = flat_hand

            # Create temporary state for cost evaluation
            temp_state = state.clone()
            temp_state.hand_states = hand_states

            # Compute energy
            energy = problem.total_energy(temp_state)  # (B,)

            # Backward for gradients
            with self._profile_section("backward"):
                energy.sum().backward()

            grad = hand_states.grad  # (B, T, D)

            return energy.detach(), grad.detach() if grad is not None else None

        finally:
            problem.context._skip_set_parameters = False

    def _sample_contacts(
        self,
        hand_model,
        context,
        current_contacts: Tensor,  # (B, n_contacts) per-trajectory contacts
        device: torch.device,
    ) -> Tensor:
        """
        Sample new contact indices with switch_possibility.

        Args:
            hand_model: The hand model
            context: Optimization context
            current_contacts: Current per-trajectory contacts, shape (B, n_contacts)
            device: Torch device

        Returns:
            New per-trajectory contacts, shape (B, n_contacts)
        """
        batch_size = current_contacts.shape[0]
        n_contact = current_contacts.shape[1]
        new_contacts = current_contacts.clone()

        # Determine which contacts to switch
        switch_mask = torch.rand(batch_size, n_contact, device=device) < self.switch_possibility

        if context.contact_sampler is not None:
            batch_switch_mask = switch_mask.any(dim=1)
            if batch_switch_mask.any():
                n_switch = batch_switch_mask.sum().item()
                new_samples = context.contact_sampler.sample(n_switch, n_contact)
                switch_indices = batch_switch_mask.nonzero(as_tuple=True)[0]
                for i, idx in enumerate(switch_indices):
                    item_switch = switch_mask[idx]
                    new_contacts[idx, item_switch] = new_samples[i, item_switch]
        else:
            # Fallback: uniform sampling
            n_candidates = hand_model.n_contact_candidates
            new_indices = torch.randint(n_candidates, size=(batch_size, n_contact), device=device)
            new_contacts[switch_mask] = new_indices[switch_mask]

        return new_contacts

    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self._ema_grad = None
        self._per_batch_step = None
        self._current_energy = None
        self._initialized = False
        self._T = None
        self._D_hand = None

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
                "T": self._T,
            }
        )
        if self._per_batch_step is not None:
            diag["mean_step"] = self._per_batch_step.float().mean().item()
        if self._current_energy is not None:
            diag["mean_energy"] = self._current_energy.mean().item()
        return diag
