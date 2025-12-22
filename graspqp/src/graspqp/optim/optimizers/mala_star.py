"""
MalaStar optimizer - reimplemented under the new optimization framework.

MALA* (Metropolis-Adjusted Langevin Algorithm Star) from GraspQP paper.
This is a gradient-based MCMC sampler with:
- RMSProp-style gradient normalization
- Temperature annealing
- Metropolis-Hastings accept/reject
- Optional contact point switching (for grasp generation mode)

IMPORTANT: This optimizer works directly with hand_model.hand_pose (like fit.py)
because set_parameters() breaks the computation graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import Normal

from .base import Optimizer

if TYPE_CHECKING:
    from ..problem import OptimizationProblem
    from ..state import TrajectoryState

normal = Normal(0, 1)


class MalaStarOptimizer(Optimizer):
    """
    MALA* optimizer from GraspQP, adapted for the new framework.

    IMPORTANT: Works directly with hand_model.hand_pose to preserve gradients.
    The state is only used for storage/export, not for gradient computation.
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

        # State variables (initialized on first step)
        self._ema_grad: Optional[Tensor] = None
        self._per_batch_step: Optional[Tensor] = None
        self._old_hand_pose: Optional[Tensor] = None
        self._old_contact_indices: Optional[Tensor] = None
        self._current_energy: Optional[Tensor] = None

    def step(
        self,
        state: "TrajectoryState",
        problem: "OptimizationProblem",
    ) -> "TrajectoryState":
        """
        Perform one MALA* optimization step.

        Works directly with hand_model.hand_pose to preserve gradients,
        matching fit.py's approach exactly.
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
        # 1. Try step: propose new parameters using gradient
        # =====================================================================
        # Ensure hand_model.hand_pose has gradient
        if hand_model.hand_pose.grad is None:
            # Need to do initial forward/backward pass
            self._compute_energy_and_grad(problem)

        # Compute step size with decay
        s = self.step_size * (self.temperature_decay ** (self._per_batch_step // self.stepsize_period))
        step_size = s.unsqueeze(-1)  # (B, 1)

        # Get gradient
        grad = hand_model.hand_pose.grad
        if grad is None:
            grad = torch.zeros_like(hand_model.hand_pose)

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
        proposed_pose = hand_model.hand_pose - step_size * normalized_grad

        # Handle NaN in proposal
        if proposed_pose.isnan().any():
            nan_mask = proposed_pose.isnan().any(dim=-1)
            proposed_pose[nan_mask] = hand_model.hand_pose[nan_mask]

        # =====================================================================
        # 2. Sample new contact indices (if not fixed)
        # =====================================================================
        if self.fix_contacts:
            contact_indices = hand_model.contact_point_indices.clone()
        else:
            contact_indices = self._sample_contacts(hand_model, problem.context, device)

        # =====================================================================
        # 3. Save old state and set new parameters
        # =====================================================================
        self._old_hand_pose = hand_model.hand_pose.detach().clone()
        self._old_contact_indices = hand_model.contact_point_indices.clone()
        old_energy = self._current_energy

        # Set new parameters
        hand_model.set_parameters(proposed_pose, contact_indices)

        # Zero grad for new computation
        if hand_model.hand_pose.grad is not None:
            hand_model.hand_pose.grad.zero_()

        # =====================================================================
        # 4. Compute new energy
        # =====================================================================
        new_energy = self._compute_energy_and_grad(problem)

        # =====================================================================
        # 5. Accept/reject using Metropolis-Hastings
        # =====================================================================
        temperature = self.starting_temperature * (
            self.temperature_decay ** (self._per_batch_step // self.annealing_period)
        )

        alpha = torch.rand(B, dtype=torch.float, device=device)
        accept = alpha < torch.exp((old_energy - new_energy) / temperature)

        # =====================================================================
        # 6. Apply accept/reject
        # =====================================================================
        with torch.no_grad():
            reject = ~accept

            # Restore rejected states
            hand_model.hand_pose[reject] = self._old_hand_pose[reject]
            hand_model.contact_point_indices[reject] = self._old_contact_indices[reject]

            # Update current energy
            self._current_energy[accept] = new_energy[accept]

            # Recompute FK for consistency
            hand_model.current_status = hand_model.fk(hand_model.hand_pose[:, 9:])

        # Increment step counter
        self._per_batch_step += 1
        self._step_count += 1

        # =====================================================================
        # 7. Return updated state (sync from hand_model)
        # =====================================================================
        final_state = state.clone()
        final_state.hand_states = hand_model.hand_pose.detach().unsqueeze(1)  # (B, 1, D_hand)

        return final_state

    def _compute_energy_and_grad(self, problem: "OptimizationProblem") -> Tensor:
        """Compute energy and backprop to get gradients on hand_model.hand_pose."""
        hand_model = problem.context.hand_model

        # Ensure requires_grad
        hand_model.hand_pose.requires_grad_(True)

        # Clear cache
        problem.context.clear_step_cache()

        # Create a temporary state from hand_model
        temp_state = problem.context.hand_model.hand_pose.unsqueeze(1)  # (B, 1, D)

        # Compute energy through problem (but hand_model already set)
        from ..state import TrajectoryState

        dummy_state = TrajectoryState(
            hand_states=temp_state,
            object_states=torch.zeros(temp_state.shape[0], 1, 7, device=temp_state.device),
        )

        # Compute energy
        energy = problem.total_energy(dummy_state)  # (B,)

        # Store for next iteration
        self._current_energy = energy.detach().clone()

        # Backward
        energy.sum().backward()

        return energy.detach()

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
        self._old_hand_pose = None
        self._old_contact_indices = None
        self._current_energy = None

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
