# Trajectory Optimizer Debugging Summary

This document summarizes the debugging session for the trajectory optimization pipeline, specifically comparing `MalaStarOptimizer` (single-frame, works) vs `MalaStarTrajectoryOptimizer` (multi-frame, gets stuck).

## Problem Statement

The goal is to optimize hand-object trajectories for video post-processing:
- **Input**: Reference trajectory with hand poses and finger contact patterns extracted from video
- **Output**: Refined trajectory with physically plausible grasps (force closure, no penetration)

## Current Status

| Optimizer | Mode | Status |
|-----------|------|--------|
| `MalaStarOptimizer` | Single frame (B samples) | ✅ Works well |
| `MalaStarTrajectoryOptimizer` | Multi-frame (B × T) | ❌ Gets stuck after ~50 steps |
| `AdamOptimizer` | Both modes | ❌ Gradient flow issues (fixed but untested) |

## Root Cause Analysis

### Why `MalaStarOptimizer` Works (Single Frame)

```python
# Direct gradient flow through hand_model.hand_pose
hand_model.hand_pose.requires_grad_(True)
grad = hand_model.hand_pose.grad

# set_parameters handles FK + contact point recomputation
hand_model.set_parameters(proposed_pose, new_contact_indices)

# Contact switching: per-contact-point probability
# Only SOME contacts change, not all at once
switch_mask = torch.rand(batch_size, n_contact) < switch_possibility
```

### Why `MalaStarTrajectoryOptimizer` Gets Stuck (Multi-Frame)

1. **Too Aggressive Contact Switching**
   ```python
   # Per-TRAJECTORY switching (all T frames, all contacts change together)
   switch_mask = torch.rand(B) < switch_possibility
   ```
   - Changing ALL contacts for ALL frames is a huge perturbation
   - New contacts likely have worse energy → rejected
   - Result: contacts rarely actually change

2. **MCMC is Wrong for Refinement**
   - MALA* was designed for **exploration** (grasp synthesis from scratch)
   - Video post-processing is **refinement** (good reference, just need fine-tuning)
   - Accept/reject mechanism prevents gradual improvement

3. **Joint Proposal Problem**
   ```
   Current: Propose (new_pose + new_contacts) → Evaluate → Accept/Reject both
   
   Problem: Even if pose improvement is good, bad contact change can cause rejection
   ```

4. **Trajectory-Level Accept/Reject**
   - Must accept/reject ALL T frames together
   - Hard to improve when any single frame has issues

## Key Code Differences

### Gradient Flow

**Single Frame (works):**
```python
# Gradient accumulates directly on hand_model.hand_pose
energy = problem.total_energy(state)
energy.sum().backward()
grad = hand_model.hand_pose.grad  # Direct access
```

**Multi-Frame (stuck):**
```python
# Gradient goes to a clone, manual FK required
hand_states = state.hand_states.clone()
hand_states.requires_grad_(True)
flat_hand = hand_states.reshape(B * T, D)

# Manual FK setup (error-prone)
hand_model.global_translation = flat_hand[:, :3]
hand_model.global_rotation = robust_compute_rotation_matrix_from_ortho6d(flat_hand[:, 3:9])
hand_model.current_status = hand_model.fk(flat_hand[:, 9:])
hand_model.hand_pose = flat_hand

# Must also recompute contact points!
hand_model.contact_points = hand_model.all_contact_points.gather(...)
```

### Contact Sampling Strategy

**Single Frame:**
```python
# Per-contact probability - fine-grained control
switch_mask = torch.rand(batch_size, n_contact) < switch_possibility
# Result: some contacts change, others stay
```

**Multi-Frame:**
```python
# Per-trajectory probability - all-or-nothing
switch_mask = torch.rand(B) < switch_possibility
# Result: either ALL contacts change or NONE
```

## Fixes Applied During Debugging

### 1. Added Warnings for Debugging

```python
# Warning when gradient is None (broken gradient flow)
if grad is None:
    warnings.warn("Gradient is None after backward pass...")
    
# Warning when NaN in EMA gradient
if self._ema_grad.isnan().any():
    warnings.warn(f"NaN detected in EMA gradient ({nan_count} values)...")
    
# Warning when NaN in proposed states
if proposed_hand.isnan().any():
    warnings.warn(f"NaN detected in proposed hand states...")
```

### 2. Fixed Prior Weight Override Bug

```python
# Before (bug): explicit --w_prior 0.0 was ignored
if args.w_prior == 0.0:
    args.w_prior = prior_weight  # Overwrites user's explicit 0.0!

# After (fixed): use None as default
parser.add_argument("--w_prior", default=None, ...)
if args.w_prior is None:
    args.w_prior = prior_weight
```

### 3. Global Best Tracking (Partial Fix)

Added tracking to preserve best solution found during optimization:
```python
if current_best < global_best_energy:
    global_best_energy = current_best
    global_best_state = state.hand_states.clone()
    global_best_contacts = hand_model.contact_point_indices.clone()

# At end: restore global best
state.hand_states = global_best_state
```

### 4. Contact Point Recomputation Fix (CRITICAL - Adam now works!)

**Problem**: When `_skip_set_parameters=True` in gradient mode, `ensure_hand_configured` did FK but didn't recompute contact points. This caused `contact_distance` cost to use stale contact positions, breaking gradient flow.

**Fix** in `graspqp/src/graspqp/optim/context.py`:
```python
if self._skip_set_parameters:
    # FK computation
    self.hand_model.global_translation = flat_hand[:, :3]
    self.hand_model.global_rotation = ...
    self.hand_model.current_status = self.hand_model.fk(flat_hand[:, 9:])
    self.hand_model.hand_pose = flat_hand
    
    # CRITICAL: Recompute contact points from FK result!
    self.hand_model.all_contact_points, self.hand_model._all_contact_normals = (
        self.hand_model.get_contact_candidates(with_normals=True)
    )
    self.hand_model.contact_points = self.hand_model.all_contact_points.gather(
        1, contact_indices.unsqueeze(-1).expand(-1, -1, 3)
    )
    self.hand_model.contact_normals = self.hand_model._all_contact_normals.gather(
        1, contact_indices.unsqueeze(-1).expand(-1, -1, 3)
    )
```

**Result**: Adam optimizer now achieves 95% energy reduction (138.50 → 6.69)!

## Proposed Solutions

### Option A: Simple Gradient Descent (Recommended)

For refinement tasks, remove MCMC entirely:

```python
class TrajectoryGradientOptimizer:
    def step(self, state, problem):
        # Pure gradient descent - no accept/reject
        energy, grad = self._compute_energy_and_grad(state, problem)
        new_states = state.hand_states - self.lr * grad
        return new_states
```

**Pros:**
- Simple, predictable behavior
- Guaranteed energy decrease (with proper learning rate)
- No stochastic rejection of good improvements

**Cons:**
- Can get stuck in local minima (but we have good reference, so OK)
- No contact exploration

### Option B: Batched Contact Initialization

Sample K contact patterns at start, run parallel optimizations:

```
1. Sample K contact patterns at initialization
2. Run B*K trajectories in parallel (each with fixed contacts)
3. Use gradient descent (no MCMC)
4. Pick best result at the end
```

**Pros:**
- Parallelizes contact exploration
- Higher success rate
- No need to switch contacts during optimization

### Option C: Two-Phase Optimization

```
Phase 1: Fix contacts, gradient descent until convergence
Phase 2: If stuck (energy plateau), sample new contacts
Phase 3: Go back to Phase 1 with new contacts
```

**Pros:**
- Decouples pose optimization from contact selection
- Matches intuitive workflow

## Key Learnings

1. **MALA* is for exploration, not refinement**
   - Accept/reject is counterproductive when you have a good starting point

2. **Contact switching should be decoupled from pose updates**
   - Don't jointly propose and evaluate (pose + contacts)
   - Fix contacts, optimize pose, then try different contacts if needed

3. **Trajectory-level accept/reject is too coarse**
   - Hard to accept when ALL T frames must improve together

4. **Reference trajectory greatly reduces search space**
   - Don't need heavy exploration when reference is given
   - Simple gradient descent should suffice

## Files Involved

- `graspqp/src/graspqp/optim/optimizers/mala_star.py` - Single-frame optimizer (reference)
- `graspqp/src/graspqp/optim/optimizers/mala_star_trajectory.py` - Multi-frame optimizer (needs redesign)
- `graspqp/src/graspqp/optim/optimizers/torch_optim.py` - Adam optimizer (gradient flow fixed)
- `scripts/hand_object_trajectory_optim.py` - Main trajectory optimization script
- `scripts/sanity_check_optim.py` - Single-frame sanity check script

## Next Steps

1. Implement **Option A** (gradient descent) as the default trajectory optimizer
2. Add **Option B** (batched contacts) for increased success rate
3. Keep MALA* trajectory optimizer for research/comparison, but not as default
