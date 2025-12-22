# Optimization Framework Design Document

> **Status**: Design Phase  
> **Created**: 2025-01-22  
> **Last Updated**: 2025-01-22

This document captures the design discussion for refactoring GraspQP's optimization framework into a generic, modular, and GPU-efficient system.

---

## Table of Contents

1. [Use Case](#use-case)
2. [Motivation](#motivation)
3. [Requirements](#requirements)
4. [Design Questions & Answers](#design-questions--answers)
5. [System Architecture](#system-architecture)
6. [GPU Efficiency Design](#gpu-efficiency-design)
7. [File Structure](#file-structure)
8. [Configuration Format](#configuration-format)
9. [Output Format](#output-format)
10. [Open Questions](#open-questions)
11. [Implementation Roadmap](#implementation-roadmap)

---

## Use Case

### Primary Application: Video Trajectory Post-Processing

**Goal**: Post-process hand+object trajectories extracted from video to make them:
- Smoother (temporal consistency)
- Physically grounded (force closure, no penetration)
- Close to the original video reference

**Scale Target**: Generate millions of frames from human video demonstrations for:
1. Training expert RL policies (as reference trajectories)
2. Training large Vision-Language-Action (VLA) models

### Input/Output

```
Input:
  - Hand pose trajectory from video: (T, D_hand) per video
  - Object pose trajectory from video: (T, D_obj) per video
  - Contact configuration (fixed for trajectory)
  - Known issues: finger-object penetration, jitter, noise

Output:
  - Optimized hand trajectory: smoother, no penetration
  - Optimized object trajectory: consistent with hand motion
  - Force closure satisfied at each frame
```

### Key Constraints

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Max trajectory length | T ≤ 30 frames | Stability, efficiency |
| Timestep | dt = 0.1s | ~3 second clips |
| Contact changes | None within trajectory | HUGE simplification |
| Batch size | B trajectories in parallel | Throughput |

### No Contact Change Assumption

This is a critical simplification that enables significant optimization:

```
┌─────────────────────────────────────────────────────────────────┐
│  Traditional: Contact can change every frame                   │
│  - Must recompute contact points each frame                    │
│  - Contact sampling is expensive                               │
│  - Force closure depends on which fingers are in contact       │
│                                                                 │
│  Simplified: Contact fixed for entire trajectory               │
│  - Contact indices computed ONCE at initialization             │
│  - Same fingers in contact for all T frames                    │
│  - Force closure uses consistent contact configuration         │
│  - Enables vectorized computation across all frames            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Motivation

The current `scripts/fit.py` implements grasp optimization with tightly coupled components. We want to refactor into a generic optimization framework that:

- Supports arbitrary cost functions
- Handles both single-frame and trajectory optimization
- Maximizes GPU utilization
- Is easy to debug and extend
- Maintains backward compatibility with existing visualization tools
- **Scales to process millions of video frames efficiently**

---

## Requirements

### Core Requirements

1. **Generic Cost Base Class**
   - `evaluate(state, context)` → energy tensor
   - `gradient(state, context)` → gradient tensor (via autograd)
   - Configurable via YAML
   - Support custom backward pass for special costs (e.g., QP solver)

2. **Trajectory Support**
   - Variable-length trajectories (T frames)
   - Physical timestep (dt in seconds)
   - Object can also move (not just hand)
   - Single frame = trajectory of length 1

3. **Temporal Costs**
   - Velocity smoothness
   - Acceleration limits
   - Temporal consistency
   - Operate on frame pairs/windows

4. **Modularity**
   - Easy to add new costs
   - Costs can share intermediate results (via cache)
   - Costs can be enabled/disabled

5. **Optimizer Abstraction**
   - Pluggable optimization algorithms
   - Wrap external libraries (torch.optim, scipy.optimize)
   - Current: MalaStar, AnnealingDexGraspNet

6. **GPU Efficiency**
   - Vectorized operations (no Python loops over frames)
   - Batched computation (T × B parallel)
   - Efficient caching
   - Memory management for large trajectories

7. **Backward Compatibility**
   - Output format compatible with `scripts/vis/visualize_result.py`
   - Extend, don't break existing format

---

## Design Questions & Answers

### Optimization Variables

| Question | Answer |
|----------|--------|
| What defines a "frame"? | Discrete timestep with physical meaning (time in seconds) |
| Can object move? | Yes, object pose can also be optimized/tracked |
| Variable-length trajectories? | Yes, T is not fixed at optimization start |

### Cost Function Inputs

| Question | Answer |
|----------|--------|
| Auxiliary info examples? | Object model, hand model, frame index, previous state, external constraints |
| Can costs share intermediate results? | Yes, via lazy cache mechanism |

### Gradient Computation

| Question | Answer |
|----------|--------|
| Gradient methods? | PyTorch autograd (keep simple for now) |
| Custom backward? | Yes, supported for special costs (e.g., QP) |

### Configuration

| Question | Answer |
|----------|--------|
| Update on the fly? | Weights changed outside cost function; config updates after loading, not during optimization |
| Add/remove costs during optimization? | No, only before start |

### Debugging

| Question | Answer |
|----------|--------|
| Key debugging features? | Per-cost energy breakdown, gradient info, profiling |
| Trajectory visualization? | Yes, separate module |

---

## System Architecture

### Core Data Structures

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TrajectoryState                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Optimization variables (what we're optimizing):                            │
│                                                                             │
│  Attributes:                                                                │
│    hand_states: Tensor[B, T, D_hand]      # B trajectories, T frames       │
│    object_states: Tensor[B, T, D_obj]     # object poses (7: pos+quat)     │
│    dt: float                               # timestep (0.1s)                │
│    device: torch.device                                                     │
│                                                                             │
│  Properties:                                                                │
│    B: int                                  # batch size (num trajectories) │
│    T: int                                  # trajectory length (≤30)       │
│    flat_hand: Tensor[B*T, D_hand]         # view for batched FK            │
│    flat_object: Tensor[B*T, D_obj]        # view for batched transforms    │
│                                                                             │
│  Methods:                                                                   │
│    get_frame(t) -> (Tensor[B, D_hand], Tensor[B, D_obj])                   │
│    velocities() -> Tensor[B, T-1, D]      # dx/dt                          │
│    accelerations() -> Tensor[B, T-2, D]   # d²x/dt²                        │
│    clone() -> TrajectoryState                                              │
│    detach() -> TrajectoryState                                             │
│    to_legacy_dict(frame=-1) -> dict       # export single frame            │
│    to_trajectory_dict() -> dict           # export full trajectory         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          ReferenceTrajectory                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  The original video trajectory (fixed, not optimized):                      │
│                                                                             │
│  Attributes:                                                                │
│    hand_states: Tensor[B, T, D_hand]      # from video extraction          │
│    object_states: Tensor[B, T, D_obj]     # from video extraction          │
│    confidence: Tensor[B, T]               # optional: per-frame confidence │
│    contact_indices: Tensor[B, n_contacts] # FIXED for entire trajectory    │
│                                                                             │
│  Note: This is the TARGET we want to stay close to while satisfying        │
│        physical constraints.                                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          OptimizationContext                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Shared context for all costs:                                              │
│                                                                             │
│  Attributes:                                                                │
│    hand_model: HandModel                                                    │
│    object_model: ObjectModel                                                │
│    reference: ReferenceTrajectory         # the video trajectory           │
│    contact_indices: Tensor[B, n_contacts] # FIXED, from reference          │
│    device: torch.device                                                     │
│    _cache: Dict[str, Tensor]                                               │
│                                                                             │
│  Methods:                                                                   │
│    get_or_compute(key, fn, scope) -> Tensor                                │
│    clear_step_cache()                                                       │
│    set_contact_indices(indices)           # set once at init               │
│                                                                             │
│  Cache scopes:                                                              │
│    - "step": cleared every optimization step                               │
│    - "trajectory": valid for entire optimization (contacts, jacobians)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cost Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CostFunction (ABC)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Attributes:                                                                │
│    name: str                               # unique identifier             │
│    weight: float                           # multiplier                    │
│    enabled: bool = True                                                     │
│    config: CostConfig                      # cost-specific params          │
│                                                                             │
│  Abstract Methods:                                                          │
│    evaluate(state: TrajectoryState, ctx: OptimizationContext) -> Tensor    │
│                                                                             │
│  Properties:                                                                │
│    is_temporal: bool                       # needs multiple frames?        │
│    required_cache_keys: List[str]                                          │
│    provided_cache_keys: List[str]                                          │
│                                                                             │
│  Class Methods:                                                             │
│    from_config(cfg: dict) -> CostFunction                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         PerFrameCost(CostFunction)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  For costs that operate on individual frames independently                  │
│                                                                             │
│  Abstract: evaluate_frame(t, state, ctx) -> Tensor[B]                      │
│  Implemented: evaluate() aggregates across frames                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        TemporalCost(CostFunction)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  For costs that need multiple frames (velocity, acceleration, etc.)        │
│                                                                             │
│  Attributes: window_size: int = 2                                          │
│  is_temporal = True                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Concrete Costs

**Reference Tracking (NEW - key for video post-processing):**
| Cost | Description | Priority |
|------|-------------|----------|
| `ReferenceTrackingCost` | E_ref: stay close to video trajectory (hand + object) | HIGH |

Note: `ReferenceTrackingCost` handles both hand and object in one cost, with configurable weights:
```yaml
reference_tracking:
  weight: 100.0
  config:
    hand_weight: 1.0       # weight for hand pose error
    object_weight: 1.0     # weight for object pose error
    # Optional: per-component weights within hand
    hand_position_weight: 10.0   # wrist position
    hand_rotation_weight: 5.0    # wrist orientation  
    finger_weight: 1.0           # finger joints
```

**Per-Frame Physical Costs (existing, to be refactored):**
| Cost | Description | Priority |
|------|-------------|----------|
| `ContactDistanceCost` | E_dis: contact points touch surface | HIGH |
| `ForceClosureCost` | E_fc: QP-based force closure | HIGH |
| `PenetrationCost` | E_pen: no hand-object penetration | HIGH |
| `SelfPenetrationCost` | E_spen: no finger self-collision | MEDIUM |
| `JointLimitCost` | E_joints: stay within joint limits | MEDIUM |

**Temporal Smoothness Costs (NEW):**
| Cost | Description | Priority |
|------|-------------|----------|
| `VelocitySmoothnessCost` | E_vel: smooth velocity | HIGH |
| `AccelerationCost` | E_acc: bounded acceleration | MEDIUM |
| `JerkCost` | E_jerk: smooth jerk (3rd derivative) | LOW |

**Optional Constraints:**
| Cost | Description | Priority |
|------|-------------|----------|
| `WallConstraintCost` | E_wall: table/obstacle avoidance | LOW |
| `WorkspaceConstraintCost` | E_ws: stay in reachable workspace | LOW |

### Cost Weighting Strategy for Video Post-Processing

```yaml
# Recommended weights for video post-processing
costs:
  # PRIMARY: Stay close to reference (the video)
  reference_tracking:
    weight: 100.0
    config:
      hand_position_weight: 10.0   # wrist position important
      hand_rotation_weight: 5.0    # wrist orientation
      finger_weight: 1.0           # fingers can deviate more
      object_weight: 10.0          # object should match video
  
  # HARD CONSTRAINTS: Physical validity
  penetration:
    weight: 1000.0                 # must not penetrate
  contact_distance:
    weight: 50.0                   # contacts should touch
  force_closure:
    weight: 10.0                   # grasp should be stable
  
  # SOFT CONSTRAINTS: Smoothness
  velocity_smoothness:
    weight: 1.0                    # smooth motion
  acceleration:
    weight: 0.1                    # bounded acceleration
```

### Optimizer Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Optimizer (ABC)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Abstract Methods:                                                          │
│    step(state, problem) -> TrajectoryState                                 │
│    reset()                                                                  │
│                                                                             │
│  Methods:                                                                   │
│    get_diagnostics() -> Dict                                               │
│                                                                             │
│  Implementations:                                                           │
│    MalaStarOptimizer          # current MALA* algorithm                    │
│    GradientDescentOptimizer   # simple GD                                  │
│    AdamOptimizer              # wraps torch.optim.Adam                     │
│    LBFGSOptimizer             # wraps torch.optim.LBFGS or scipy           │
│    ScipyOptimizer             # wraps scipy.optimize                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Problem & Runner

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OptimizationProblem                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Attributes:                                                                │
│    context: OptimizationContext                                             │
│    costs: OrderedDict[str, CostFunction]                                   │
│    weight_schedule: Dict[str, WeightSchedule]                              │
│                                                                             │
│  Methods:                                                                   │
│    add_cost(cost)                                                          │
│    remove_cost(name)                                                       │
│    set_weight(name, weight)                                                │
│    total_energy(state) -> Tensor[B]                                        │
│    evaluate_all(state) -> Dict[str, Tensor]                                │
│    gradient(state) -> TrajectoryState                                      │
│                                                                             │
│  Class Methods:                                                             │
│    from_config(path, ctx) -> OptimizationProblem                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          OptimizationRunner                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Attributes:                                                                │
│    problem: OptimizationProblem                                             │
│    optimizer: Optimizer                                                     │
│    callbacks: List[Callback]                                                │
│    state: TrajectoryState                                                   │
│                                                                             │
│  Methods:                                                                   │
│    run(n_iters) -> TrajectoryState                                         │
│    step() -> TrajectoryState                                               │
│                                                                             │
│  Callbacks:                                                                 │
│    LoggingCallback, WandbCallback, CheckpointCallback,                     │
│    ProfilerCallback, EarlyStoppingCallback, ResetCallback                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## GPU Efficiency Design

### Tensor Layout

```
Recommended: (B, T, D) - batch first, time contiguous per trajectory

hand_states: Tensor[B, T, D_hand]    # B trajectories, T frames each
object_states: Tensor[B, T, D_obj]   # object poses (also optimized)

┌─────────────────────────────────────────────────────────────────┐
│  Trajectory 0: [frame_0, frame_1, ..., frame_T-1]  ─┐          │
│  Trajectory 1: [frame_0, frame_1, ..., frame_T-1]   │ Batch    │
│  ...                                                │ parallel │
│  Trajectory B-1: [frame_0, frame_1, ..., frame_T-1]─┘          │
└─────────────────────────────────────────────────────────────────┘

Why (B, T, D) over (T, B, D)?
- Temporal operations along dim=1 are natural: x[:, 1:] - x[:, :-1]
- Each trajectory is contiguous in memory
- Flatten to (B*T, D) for batched FK/SDF operations
```

### Batched Trajectory Processing

```python
# B=64 trajectories, T=30 frames, D=25 dimensions
hand_states: Tensor[64, 30, 25]

# For per-frame costs (FK, SDF): flatten to (B*T, D)
flat = hand_states.reshape(B * T, D)        # (1920, 25)
hand_model.set_parameters(flat)              # single batched FK call
contact_pts = hand_model.contact_points      # (1920, n_contacts, 3)
sdf = object_model.cal_distance(contact_pts) # single batched SDF call
sdf = sdf.reshape(B, T, n_contacts)          # back to trajectory shape

# For temporal costs: operate along dim=1
velocity = hand_states[:, 1:] - hand_states[:, :-1]  # (B, T-1, D)
```

### Fixed Contact Optimization

Since contacts don't change within a trajectory:

```python
class FixedContactContext:
    def __init__(self, hand_model, contact_indices):
        # Contact indices: (B, n_contacts) - same for all T frames
        self.contact_indices = contact_indices  # computed ONCE
        
        # Pre-compute contact Jacobians if needed
        self.contact_jacobians = None  # lazy compute, cache for trajectory
    
    def get_contact_points(self, hand_states):
        # hand_states: (B, T, D)
        flat = hand_states.reshape(-1, D)
        hand_model.set_parameters(flat, contact_point_indices=self.contact_indices)
        pts = hand_model.contact_points  # (B*T, n_contacts, 3)
        return pts.reshape(B, T, n_contacts, 3)
```

### Cost Evaluation Strategies

**1. Fully Vectorized (all B×T at once)**
```python
class PenetrationCost(PerFrameCost):
    def evaluate(self, state: TrajectoryState, ctx) -> Tensor[B]:
        # Process all B*T configurations in one kernel
        flat = state.hand_states.reshape(-1, D)      # (B*T, D)
        ctx.hand_model.set_parameters(flat)
        
        # Batched SDF query
        surface_pts = ctx.hand_model.surface_points  # (B*T, n_pts, 3)
        sdf = ctx.object_model.cal_distance(surface_pts)[0]  # (B*T, n_pts)
        
        # Penetration = negative SDF
        pen = F.relu(-sdf).sum(dim=-1)               # (B*T,)
        pen = pen.reshape(state.B, state.T)          # (B, T)
        
        # Sum across time (or could be max, mean)
        return pen.sum(dim=1)                        # (B,)
```

**2. Temporal Vectorized**
```python
class VelocitySmoothnessCost(TemporalCost):
    def evaluate(self, state: TrajectoryState, ctx) -> Tensor[B]:
        # Vectorized velocity: (B, T-1, D)
        vel = state.hand_states[:, 1:] - state.hand_states[:, :-1]
        vel = vel / state.dt  # actual velocity
        
        # L2 norm squared, sum over time
        return (vel ** 2).sum(dim=(1, 2))  # (B,)

class AccelerationCost(TemporalCost):
    def evaluate(self, state: TrajectoryState, ctx) -> Tensor[B]:
        vel = state.hand_states[:, 1:] - state.hand_states[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]  # (B, T-2, D)
        return (acc ** 2).sum(dim=(1, 2))  # (B,)
```

**3. Reference Tracking (key for video post-processing)**
```python
class ReferenceTrackingCost(PerFrameCost):
    """Stay close to the original video trajectory"""
    
    def __init__(self, reference_hand: Tensor, reference_object: Tensor, ...):
        self.ref_hand = reference_hand      # (B, T, D_hand)
        self.ref_object = reference_object  # (B, T, D_obj)
    
    def evaluate(self, state: TrajectoryState, ctx) -> Tensor[B]:
        # Weighted distance to reference
        hand_error = (state.hand_states - self.ref_hand) ** 2
        obj_error = (state.object_states - self.ref_object) ** 2
        
        # Could weight different components differently
        # e.g., higher weight on wrist position, lower on fingers
        return (hand_error.sum(dim=(1, 2)) + 
                self.object_weight * obj_error.sum(dim=(1, 2)))
```

### Cache Design

```python
class OptimizationContext:
    def __init__(self, ...):
        self._cache = {}
        self._contact_indices = None  # FIXED for entire trajectory
    
    def set_fixed_contacts(self, contact_indices: Tensor):
        """Set once at initialization, used for all frames"""
        self._contact_indices = contact_indices  # (B, n_contacts)
    
    def get_or_compute(self, key, compute_fn, scope="step"):
        """
        scope options:
        - "step": cleared every optimization step
        - "trajectory": valid for entire trajectory (e.g., contact indices)
        - "persistent": never cleared
        """
```

### Memory Budget

For B=64 trajectories, T=30 frames:

| Component | Size | Memory |
|-----------|------|--------|
| hand_states (B, T, D=25) | 64×30×25 | ~200 KB |
| object_states (B, T, D=7) | 64×30×7 | ~50 KB |
| contact_points (B×T, n_c=8, 3) | 1920×8×3 | ~180 KB |
| SDF values (B×T, n_pts=1000) | 1920×1000 | ~7.5 MB |
| Gradients (same as forward) | ~2× | ~15 MB |

**Total: ~25-50 MB per batch** - very manageable!

---

## File Structure

```
graspqp/
├── src/graspqp/
│   ├── optim/                          # NEW: optimization framework
│   │   ├── __init__.py
│   │   ├── state.py                    # TrajectoryState, FrameState
│   │   ├── context.py                  # OptimizationContext
│   │   ├── problem.py                  # OptimizationProblem
│   │   ├── runner.py                   # OptimizationRunner
│   │   │
│   │   ├── costs/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # CostFunction, PerFrameCost, TemporalCost
│   │   │   ├── contact.py              # ContactDistanceCost
│   │   │   ├── force_closure.py        # ForceClosureCost
│   │   │   ├── penetration.py          # PenetrationCost, SelfPenetrationCost
│   │   │   ├── prior.py                # PriorPoseCost
│   │   │   ├── constraints.py          # JointLimitCost, WallConstraintCost
│   │   │   ├── temporal.py             # VelocitySmoothness, Acceleration
│   │   │   └── registry.py             # Cost registration for YAML
│   │   │
│   │   ├── optimizers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # Optimizer ABC
│   │   │   ├── mala_star.py            # MalaStarOptimizer
│   │   │   ├── gradient.py             # GradientDescentOptimizer
│   │   │   ├── torch_optim.py          # AdamOptimizer, LBFGSOptimizer
│   │   │   └── scipy_optim.py          # ScipyOptimizer
│   │   │
│   │   └── callbacks/
│   │       ├── __init__.py
│   │       ├── base.py                 # Callback ABC
│   │       ├── logging.py
│   │       ├── wandb.py
│   │       ├── checkpoint.py
│   │       └── profiler.py
│   │
│   ├── core/                           # existing, minimal changes
│   └── ...
```

---

## Configuration Format

```yaml
# configs/grasp_optimization.yaml

problem:
  trajectory:
    n_frames: 1              # 1 = single frame (current behavior)
    dt: 0.01                 # timestep in seconds
  
  costs:
    - name: contact_distance
      type: ContactDistanceCost
      weight: 100.0
      config:
        reduction: sum
    
    - name: force_closure
      type: ForceClosureCost
      weight: 1.0
      config:
        friction: 0.2
        n_cone_vecs: 4
        solver: qp
    
    - name: penetration
      type: PenetrationCost
      weight: 100.0
    
    - name: self_penetration
      type: SelfPenetrationCost
      weight: 10.0
    
    - name: velocity_smoothness
      type: VelocitySmoothnessCost
      weight: 1.0
      enabled: false         # only for trajectory

optimizer:
  type: MalaStar
  config:
    step_size: 0.005
    temperature: 18.0
    temperature_decay: 0.95
    annealing_period: 30

runner:
  n_iters: 7000
  callbacks:
    - type: LoggingCallback
      config:
        log_every: 100
    - type: CheckpointCallback
      config:
        save_every: 500
```

---

## Output Format

### Single Frame (T=1) - Backward Compatible

For single-frame optimization (original grasp synthesis), output is unchanged:

```python
{
    "values": Tensor(B,),
    "parameters": {
        "root_pose": Tensor(B, 7),
        "joint_0": Tensor(B,),
        ...
    },
    "grasp_velocities": {...},
    "contact_idx": Tensor(B, n_contacts),
    "grasp_type": str,
    "contact_links": List[str],
}
```

### Trajectory (T>1) - Extended for Video Post-Processing

```python
{
    # === LEGACY FIELDS (last frame, for backward compat with visualize_result.py) ===
    "values": Tensor(B,),             # total energy per trajectory
    "parameters": {...},              # LAST frame hand pose
    "grasp_velocities": {...},        # LAST frame
    "contact_idx": Tensor(B, n_contacts),  # fixed for trajectory
    "grasp_type": str,
    "contact_links": List[str],
    
    # === TRAJECTORY DATA ===
    "trajectory": {
        "n_frames": int,              # T
        "dt": float,                  # timestep (0.1s)
        "hand_states": Tensor(B, T, D_hand),    # optimized hand trajectory
        "object_states": Tensor(B, T, D_obj),   # optimized object trajectory
    },
    
    # === REFERENCE (original video) ===
    "reference": {
        "hand_states": Tensor(B, T, D_hand),    # original from video
        "object_states": Tensor(B, T, D_obj),   # original from video
    },
    
    # === DIAGNOSTICS ===
    "cost_breakdown": {
        "reference_tracking": Tensor(B,),
        "penetration": Tensor(B,),
        "velocity_smoothness": Tensor(B,),
        "force_closure": Tensor(B,),
        ...
    },
    "per_frame_energy": Tensor(B, T),  # energy at each frame
    
    # === METADATA ===
    "metadata": {
        "optimizer": str,
        "n_iters": int,
        "config_file": str,
        "timestamp": str,
        "source_video": str,          # optional: source video path
    }
}
```

### Per-Trajectory vs Per-Batch Files

For large-scale processing:

```
Option A: One file per batch (current approach extended)
  output/batch_001.pt  # contains B trajectories, each with T frames

Option B: One file per trajectory (simpler for downstream)
  output/traj_00001.pt  # single trajectory, T frames
  output/traj_00002.pt
  ...
```

Recommendation: Start with Option A (batched), add Option B export if needed.

---

## Open Questions

### Answered

| # | Question | Answer |
|---|----------|--------|
| Q9 | Is object pose an optimization variable or given input? | **(A) Optimized** - both hand and object are optimized |
| Q10 | Temporal cost window? | **(A) Pairs** - consecutive frames (t, t+1) |
| Q11 | Primary use case for trajectory? | **Video post-processing** for RL/VLA training |
| Q12 | Long trajectory handling? | **(B) Hard limit** - T ≤ 30, break longer videos |
| Q13 | Cost fusion strategy? | **(A) Fully modular** initially, optimize later |
| Q14 | Auto profiling suggestions? | **Yes** - helpful for scale |

### Additional Answers (2025-01-22)

| # | Question | Answer |
|---|----------|--------|
| Q15 | Contact detection from video? | **Video reference specifies contacts** - which fingers are in contact is part of the input |
| Q16 | Object model source? | **Mesh is given** - already optimized/simplified for efficient SDF computation |
| Q17 | Hand representation? | **Support both MANO (human) and robot hands (Allegro, etc.)** |
| Q18 | Batch loading strategy? | **Define data interface now** - no existing pipeline, design clean API |
| Q19 | Convergence criteria? | **Flexible/customizable**: fixed iterations, energy threshold, early stopping (success or failure) |
| Q20 | Variable-length trajectories? | **Padding + masking** - pad to max length in batch, use mask for valid frames |
| Q21 | Multiple optimization runs? | **Yes, K perturbations per trajectory** - increases chance of finding valid solution |

---

## Implementation Roadmap

### Phase 1: Core Framework with Trajectory Support
> Goal: Get basic video post-processing working end-to-end

- [ ] `TrajectoryState` with (B, T, D) layout
- [ ] `ReferenceTrajectory` to hold video data
- [ ] `OptimizationContext` with fixed contacts
- [ ] `CostFunction` base class (PerFrameCost, TemporalCost)
- [ ] Core costs:
  - [ ] `ReferenceTrackingCost` (stay close to video)
  - [ ] `PenetrationCost` (no hand-object penetration)
  - [ ] `VelocitySmoothnessCost` (smooth motion)
- [ ] `OptimizationProblem`
- [ ] `AdamOptimizer` (simple, effective for this use case)
- [ ] `OptimizationRunner` with basic callbacks
- [ ] YAML configuration loading
- [ ] Output format (backward compatible + trajectory)

### Phase 2: Physical Constraints
> Goal: Add force closure and contact constraints

- [ ] `ContactDistanceCost` (contacts touch surface)
- [ ] `ForceClosureCost` (stable grasp)
- [ ] `SelfPenetrationCost` (no finger collision)
- [ ] `JointLimitCost` (valid joint angles)

### Phase 3: Efficiency & Scale
> Goal: Process millions of frames efficiently

- [ ] Batch loading of video trajectories
- [ ] Multi-GPU support (DataParallel)
- [ ] Mixed precision (fp16)
- [ ] Profiling and bottleneck identification
- [ ] Checkpoint/resume for large datasets

### Phase 4: Advanced Features
- [ ] `MalaStarOptimizer` (for comparison)
- [ ] `LBFGSOptimizer` (potentially faster convergence)
- [ ] Weight scheduling
- [ ] Confidence-weighted reference tracking
- [ ] Trajectory visualization module

---

## Batch Processing at Scale

### Variable-Length Trajectories

Millions of trajectories with different lengths (T varies, max 30):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Padding + Masking                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Batch of B trajectories, pad to T_max:                                    │
│                                                                             │
│  hand_states: Tensor[B, T_max, D_hand]                                     │
│  valid_mask:  Tensor[B, T_max]  # 1 for valid, 0 for padding               │
│  lengths:     Tensor[B]         # actual length of each trajectory         │
│                                                                             │
│  Example (T_max=8):                                                        │
│  Traj 0: [f0, f1, f2, f3, f4, PAD, PAD, PAD]  mask=[1,1,1,1,1,0,0,0]       │
│  Traj 1: [f0, f1, f2, f3, f4, f5, f6, f7]     mask=[1,1,1,1,1,1,1,1]       │
│  Traj 2: [f0, f1, f2, PAD, PAD, PAD, PAD, PAD] mask=[1,1,1,0,0,0,0,0]      │
│                                                                             │
│  Masked energy computation:                                                │
│    per_frame_energy: (B, T_max)                                            │
│    masked_energy = (per_frame_energy * valid_mask).sum(dim=1)              │
│    normalized = masked_energy / lengths  # optional: per-frame average    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Perturbation Optimization

Run K perturbations per trajectory to increase success rate:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Two-Level Batching: (B, K, T, D)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  B = number of different video trajectories (e.g., 64)                     │
│  K = perturbation attempts per trajectory (e.g., 8)                        │
│  T = trajectory length (padded to T_max)                                   │
│  D = state dimension                                                       │
│                                                                             │
│  Reference: (B, T, D)      ← original video, no K dimension                │
│  Initial:   (B, K, T, D)   ← K perturbed copies of each reference          │
│  Optimized: (B, K, T, D)   ← after optimization                            │
│                                                                             │
│  For computation: flatten to (B*K, T, D) = (512, 30, D)                    │
│  After optimization: reshape back to (B, K, T, D)                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Perturbation generation:
┌─────────────────────────────────────────────────────────────────────────────┐
│  reference: (B, T, D)                                                      │
│  noise = torch.randn(B, K, T, D) * perturbation_scale                     │
│  initial = reference.unsqueeze(1) + noise  # (B, K, T, D)                 │
│                                                                             │
│  Perturbation strategies:                                                  │
│  - Gaussian noise (simple)                                                 │
│  - Structured noise (more on position, less on fingers)                   │
│  - Temporal smoothing of noise                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Result selection:
┌─────────────────────────────────────────────────────────────────────────────┐
│  After optimization:                                                       │
│  energies: (B, K)          ← total energy per trajectory per perturbation │
│  valid: (B, K)             ← binary mask: constraints satisfied?          │
│                                                                             │
│  Selection strategies:                                                     │
│  1. Best valid: argmin(energies) where valid==True                        │
│  2. Best overall: argmin(energies) regardless of validity                 │
│  3. All valid: return all (B, K) that satisfy constraints                 │
│  4. Failure: if no K attempts valid, mark trajectory as unfixable         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Updated TrajectoryState

```python
class TrajectoryState:
    """Supports both single and multi-perturbation optimization"""
    
    def __init__(
        self,
        hand_states: Tensor,       # (B, T, D) or (B, K, T, D)
        object_states: Tensor,     # (B, T, D) or (B, K, T, D)
        valid_mask: Tensor,        # (B, T) or (B, K, T)
        dt: float = 0.1,
    ):
        self.hand_states = hand_states
        self.object_states = object_states
        self.valid_mask = valid_mask
        self.dt = dt
        
        # Detect layout
        if hand_states.dim() == 3:
            self.B, self.T, self.D_hand = hand_states.shape
            self.K = 1
            self._has_perturbations = False
        else:
            self.B, self.K, self.T, self.D_hand = hand_states.shape
            self._has_perturbations = True
    
    @property
    def flat(self) -> "TrajectoryState":
        """Flatten (B, K, T, D) to (B*K, T, D) for computation"""
        if not self._has_perturbations:
            return self
        return TrajectoryState(
            hand_states=self.hand_states.reshape(self.B * self.K, self.T, -1),
            object_states=self.object_states.reshape(self.B * self.K, self.T, -1),
            valid_mask=self.valid_mask.reshape(self.B * self.K, self.T),
            dt=self.dt,
        )
    
    def unflatten(self, B: int, K: int) -> "TrajectoryState":
        """Reshape (B*K, T, D) back to (B, K, T, D)"""
        return TrajectoryState(
            hand_states=self.hand_states.reshape(B, K, self.T, -1),
            object_states=self.object_states.reshape(B, K, self.T, -1),
            valid_mask=self.valid_mask.reshape(B, K, self.T),
            dt=self.dt,
        )
    
    @staticmethod
    def from_reference(
        reference: "ReferenceTrajectory",
        n_perturbations: int = 1,
        perturbation_scale: float = 0.01,
    ) -> "TrajectoryState":
        """Create initial state with K perturbations from reference"""
        B, T, D_hand = reference.hand_states.shape
        K = n_perturbations
        
        if K == 1:
            return TrajectoryState(
                hand_states=reference.hand_states.clone(),
                object_states=reference.object_states.clone(),
                valid_mask=reference.valid_mask.clone(),
            )
        
        # Generate K perturbations
        hand_noise = torch.randn(B, K, T, D_hand) * perturbation_scale
        obj_noise = torch.randn(B, K, T, reference.object_states.shape[-1]) * perturbation_scale
        
        return TrajectoryState(
            hand_states=reference.hand_states.unsqueeze(1) + hand_noise,
            object_states=reference.object_states.unsqueeze(1) + obj_noise,
            valid_mask=reference.valid_mask.unsqueeze(1).expand(B, K, T),
        )
```

### Result Selection

```python
class ResultSelector:
    """Select best results from multi-perturbation optimization"""
    
    @staticmethod
    def select_best_valid(
        state: TrajectoryState,      # (B, K, T, D)
        energies: Tensor,            # (B, K)
        valid: Tensor,               # (B, K) bool
    ) -> Tuple[TrajectoryState, Tensor, Tensor]:
        """
        Returns:
          - best_state: (B, T, D) - best valid trajectory per batch
          - best_energy: (B,) - energy of selected trajectory
          - success: (B,) bool - whether any valid solution found
        """
        # Mask invalid with large energy
        masked_energies = energies.clone()
        masked_energies[~valid] = float('inf')
        
        # Find best per batch
        best_k = masked_energies.argmin(dim=1)  # (B,)
        success = valid.any(dim=1)               # (B,)
        
        # Gather best trajectories
        B = state.B
        best_hand = state.hand_states[torch.arange(B), best_k]      # (B, T, D)
        best_object = state.object_states[torch.arange(B), best_k]  # (B, T, D)
        best_energy = energies[torch.arange(B), best_k]             # (B,)
        
        return (
            TrajectoryState(best_hand, best_object, state.valid_mask[:, 0]),
            best_energy,
            success,
        )
```

---

## Data Interface

### Input Format (Video Reference)

```python
# Each video clip produces one ReferenceTrajectory
# Multiple clips can be batched (B trajectories)

reference_data = {
    # Required fields - PRE-TRANSFORMED COORDINATES
    # Hand pose is RELATIVE to object (Hand_T_Object)
    "hand_states": Tensor[B, T, D_hand],    
    
    # Object pose is GLOBAL (Object_T_World)
    "object_states": Tensor[B, T, D_obj],   
    
    "contact_indices": Tensor[B, n_contacts],  # which fingers in contact (FIXED)
    
    # Hand configuration
    "hand_type": str,                        # "mano" | "allegro" | "leap" | ...
    "hand_side": str,                        # "left" | "right" (for MANO)
    
    # Object configuration  
    "object_mesh_path": str,                 # path to simplified mesh
    "object_scale": float,                   # mesh scale factor
    
    # Optional fields
    "confidence": Tensor[B, T],              # per-frame confidence from video
    "contact_links": List[str],              # which links are in contact
    "dt": float,                             # timestep (default 0.1s)
    
    # Metadata
    "source_video": str,                     # source video path
    "frame_indices": Tensor[B, T],           # original frame numbers
}
```

### Coordinate Systems (Critical Optimization)

To decouple physics from global motion:

1.  **Optimization Variables**:
    *   `hand_states`: **Hand_T_Object** (Pose of hand in Object frame)
    *   `object_states`: **Object_T_World** (Pose of object in World frame)

2.  **Benefit**:
    *   Physics costs (penetration, contact, stability) depend ONLY on `hand_states`.
    *   Object SDF can be cached in canonical object frame (never transformed).
    *   Global trajectory smoothness depends on `object_states`.
    *   Optimizing `hand_states` is more stable as values are smaller/local.

3.  **Forward Kinematics**:
    *   Compute hand points in Object frame directly.
    *   For visualization/export: `Hand_T_World = Object_T_World @ Hand_T_Object`.

### Hand State Format

```python
# MANO (human hand): D_hand = 51
mano_state = {
    "global_orient": Tensor[3],      # axis-angle rotation
    "transl": Tensor[3],             # wrist translation
    "hand_pose": Tensor[45],         # 15 joints × 3 axis-angle
}
# Flattened: [transl(3), global_orient(3), hand_pose(45)] = 51

# Robot hand (Allegro): D_hand = 22
allegro_state = {
    "root_pos": Tensor[3],           # wrist position
    "root_quat": Tensor[4],          # wrist quaternion (wxyz)
    "joint_pos": Tensor[16],         # 16 joint angles
}
# Flattened: [root_pos(3), root_quat(4), joint_pos(16)] = 23
# Or with 6D rotation: [root_pos(3), root_rot6d(6), joint_pos(16)] = 25
```

### Object State Format

```python
# D_obj = 7 (position + quaternion)
object_state = {
    "position": Tensor[3],           # object center position
    "quaternion": Tensor[4],         # orientation (wxyz)
}
# Flattened: [position(3), quaternion(4)] = 7
```

### Contact Specification

```python
# Contact indices: which contact candidates are active
# Shape: (B, n_contacts) where n_contacts is fixed (e.g., 8)

# For MANO: indices into fingertip vertices
contact_indices = Tensor[[745, 320, 443, 555, 672, 789, 890, 123]]

# For Allegro: indices into contact candidate list
contact_indices = Tensor[[0, 5, 12, 18, 25, 32, 40, 48]]

# Contact links (optional): which links are in contact
contact_links = ["index_tip", "middle_tip", "ring_tip", "thumb_tip"]
```

---

## Convergence Criteria

### Supported Stopping Conditions

```python
class StoppingCondition(ABC):
    @abstractmethod
    def should_stop(self, step: int, state: TrajectoryState, 
                    energies: Dict[str, Tensor]) -> Tuple[bool, str]:
        """Returns (should_stop, reason)"""
        pass

class MaxIterations(StoppingCondition):
    """Stop after fixed number of iterations"""
    def __init__(self, max_iters: int):
        self.max_iters = max_iters

class EnergyThreshold(StoppingCondition):
    """Stop when total energy below threshold"""
    def __init__(self, threshold: float):
        self.threshold = threshold

class EnergyPlateau(StoppingCondition):
    """Stop when energy stops improving"""
    def __init__(self, patience: int = 100, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta

class CostFailure(StoppingCondition):
    """Stop early if specific cost exceeds failure threshold"""
    def __init__(self, cost_name: str, threshold: float):
        # e.g., stop if penetration > 0.01 (can't fix)
        self.cost_name = cost_name
        self.threshold = threshold

class CostSuccess(StoppingCondition):
    """Stop early if all costs below success thresholds"""
    def __init__(self, thresholds: Dict[str, float]):
        # e.g., {"penetration": 0.001, "force_closure": 0.1}
        self.thresholds = thresholds
```

### Combining Conditions

```python
# In YAML config:
stopping:
  conditions:
    - type: MaxIterations
      config:
        max_iters: 2000
    
    - type: CostSuccess
      config:
        thresholds:
          penetration: 0.001
          force_closure: 0.1
          reference_tracking: 1.0
    
    - type: CostFailure
      config:
        cost_name: penetration
        threshold: 0.1  # give up if penetration still > 0.1
    
    - type: EnergyPlateau
      config:
        patience: 200
        min_delta: 1e-5
  
  # How to combine: "any" (stop on first) or "all" (need all)
  mode: any
```

---

## Hand Model Abstraction

To support both MANO and robot hands:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HandModelInterface (ABC)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Properties:                                                                │
│    n_joints: int                           # number of actuated joints     │
│    n_contact_candidates: int               # total contact points          │
│    state_dim: int                          # dimension of hand state       │
│                                                                             │
│  Abstract Methods:                                                          │
│    set_state(state: Tensor[B, D])          # set hand configuration        │
│    get_contact_points(indices) -> Tensor   # get contact point positions   │
│    get_surface_points() -> Tensor          # for penetration check         │
│    forward_kinematics() -> Dict            # compute all link transforms   │
│                                                                             │
│  Implementations:                                                           │
│    AllegroHandModel                        # existing                      │
│    LeapHandModel                           # existing                      │
│    MANOHandModel                           # NEW: wraps MANO               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MANO Integration

```python
class MANOHandModel(HandModelInterface):
    def __init__(self, mano_model_path: str, side: str = "right"):
        self.mano = MANO(model_path=mano_model_path, is_rhand=(side == "right"))
        
    def set_state(self, state: Tensor):
        # state: (B, 51) = [transl(3), global_orient(3), hand_pose(45)]
        transl = state[:, :3]
        global_orient = state[:, 3:6]
        hand_pose = state[:, 6:51]
        
        output = self.mano(
            global_orient=global_orient,
            hand_pose=hand_pose,
            transl=transl,
        )
        self.vertices = output.vertices  # (B, 778, 3)
        self.joints = output.joints      # (B, 21, 3)
    
    def get_contact_points(self, indices: Tensor) -> Tensor:
        # indices: (B, n_contacts) - vertex indices
        return self.vertices.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))
```

---

## Example Usage (Target API)

```python
from graspqp.optim import (
    TrajectoryState, ReferenceTrajectory, OptimizationContext,
    OptimizationProblem, OptimizationRunner, AdamOptimizer
)
from graspqp.optim.costs import (
    ReferenceTrackingCost, PenetrationCost, VelocitySmoothnessCost,
    ForceClosureCost, ContactDistanceCost
)

# Load video trajectories (e.g., from HaMeR, HAMER, etc.)
video_hand_poses = load_video_hand_poses("video_001.pt")   # (B, T, D_hand)
video_object_poses = load_video_object_poses("video_001.pt")  # (B, T, D_obj)
contact_indices = detect_contacts(video_hand_poses, video_object_poses)  # (B, n_contacts)

# Create reference (the video trajectory)
reference = ReferenceTrajectory(
    hand_states=video_hand_poses,
    object_states=video_object_poses,
    contact_indices=contact_indices,
)

# Initialize optimization state (start from reference)
state = TrajectoryState(
    hand_states=video_hand_poses.clone(),
    object_states=video_object_poses.clone(),
    dt=0.1,
)

# Create context
ctx = OptimizationContext(
    hand_model=get_hand_model("allegro", device),
    object_model=ObjectModel(...),
    reference=reference,
)

# Build problem
problem = OptimizationProblem(ctx)
problem.add_cost(ReferenceTrackingCost(weight=100.0))
problem.add_cost(PenetrationCost(weight=1000.0))
problem.add_cost(VelocitySmoothnessCost(weight=1.0))
problem.add_cost(ForceClosureCost(weight=10.0))
problem.add_cost(ContactDistanceCost(weight=50.0))

# Or load from YAML
# problem = OptimizationProblem.from_config("configs/video_postprocess.yaml", ctx)

# Create optimizer and runner
optimizer = AdamOptimizer(lr=0.01)
runner = OptimizationRunner(problem, optimizer, callbacks=[
    LoggingCallback(log_every=100),
    CheckpointCallback(save_every=500),
])

# Run optimization
optimized_state = runner.run(n_iters=2000, initial_state=state)

# Export results
optimized_state.to_trajectory_dict("output/optimized_001.pt")
```

### 2025-01-22: Efficiency & Robustness Review

**Key Optimizations Identified:**

1.  **Remove QP Bottleneck**: 
    - QP solver (ForceClosure) is too slow for 15k+ instances per step.
    - **Solution**: Use `GeometricStabilityCost` (SVD of Grasp Matrix + Normal alignment) as O(1) proxy in main loop. Run QP only for final validation.

2.  **"Hunger Games" Pruning**:
    - Don't waste compute on diverging perturbations.
    - **Solution**: Start with high K (e.g., 32), prune to K=8, then K=1 over time based on energy.

3.  **Disentangled Coordinates (Adopted immediately)**:
    - Optimization variables should be `Object_T_World` and `Hand_T_Object`.
    - **Benefit**: Physics costs (penetration/contact) become invariant to global motion; SDF caching is simpler.

4.  **Capsule Proxy for Penetration**:
    - 778 vertices per hand is expensive.
    - **Solution**: Use ~20 capsules/spheres for "coarse" penetration cost, much faster.

5.  **Contact Stationarity**:
    - Fixed indices != fixed position. Fingers could slide.
    - **Solution**: Add `ContactStationarityCost` to penalize sliding relative to object surface.

6.  **Curriculum Learning**:
    - Bad initial penetration causes gradient explosions.
    - **Solution**: Schedule weights (Penetration high → Reference high) to "pop" hand out then refine.

---

## Implementation Roadmap

### Phase 1: Core Framework with Trajectory Support
> Goal: Get basic video post-processing working end-to-end

- [ ] `TrajectoryState` with (B, T, D) layout
- [ ] **Coordinate System**: Implement `hand_T_object` and `object_T_world` variables
- [ ] `ReferenceTrajectory` to hold video data
- [ ] `OptimizationContext` with fixed contacts
- [ ] `CostFunction` base class (PerFrameCost, TemporalCost)
- [ ] Core costs:
  - [ ] `ReferenceTrackingCost` (stay close to video)
  - [ ] `PenetrationCost` (no hand-object penetration)
  - [ ] `VelocitySmoothnessCost` (smooth motion)
- [ ] `OptimizationProblem`
- [ ] `AdamOptimizer` (simple, effective for this use case)
- [ ] `OptimizationRunner` with basic callbacks
- [ ] YAML configuration loading
- [ ] Output format (backward compatible + trajectory)

### Phase 2: Physical Constraints & Stability (Optimized)
> Goal: Add stability without QP bottleneck

- [ ] `ContactDistanceCost` (contacts touch surface)
- [ ] **`GeometricStabilityCost`** (SVD-based proxy for force closure)
- [ ] **`ContactStationarityCost`** (prevent sliding on object)
- [ ] `SelfPenetrationCost` (no finger collision)
- [ ] `JointLimitCost` (valid joint angles)
- [ ] **QP Validation**: Run expensive QP check only at end/checkpoints

### Phase 3: Efficiency & Scale
> Goal: Process millions of frames efficiently

- [ ] **PruningCallback**: Implement "Hunger Games" strategy (K=32 → K=1)
- [ ] **Capsule Proxies**: Fast approximation for hand model
- [ ] Batch loading of video trajectories
- [ ] Multi-GPU support (DataParallel)
- [ ] Mixed precision (fp16)
- [ ] Profiling and bottleneck identification
- [ ] Checkpoint/resume for large datasets

### Phase 4: Advanced Features
- [ ] Weight scheduling (Curriculum)
- [ ] `MalaStarOptimizer` (for comparison)
- [ ] `LBFGSOptimizer` (potentially faster convergence)
- [ ] Confidence-weighted reference tracking
- [ ] Trajectory visualization module

