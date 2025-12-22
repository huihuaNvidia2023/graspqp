# Optimization Framework

This module provides a modular optimization framework for grasp and trajectory optimization.

## Sanity Check

To verify the optimization framework works correctly, run the sanity check script which mirrors the behavior of `scripts/fit.py` exactly.

### Running the Sanity Check

```bash
# Activate the graspqp conda environment first
conda activate graspqp

# Run the sanity check (uses same MalaStar optimizer as fit.py)
python scripts/sanity_check_optim.py \
    --data_root_path ./objects \
    --object_code_list apple \
    --hand_name allegro \
    --batch_size 16 \
    --n_iter 500 \
    --prior_file configs/extracted_prior.yaml \
    --w_prior 100
```

### Comparing with fit.py

Run `fit.py` with the same parameters to compare results:

```bash
python scripts/fit.py \
    --data_root_path ./objects \
    --object_code_list apple \
    --hand_name allegro \
    --batch_size 16 \
    --n_iter 500 \
    --prior_file configs/extracted_prior.yaml \
    --w_prior 100
```

### Expected Output

Both scripts should show:
- Energy decreasing over iterations
- Similar energy breakdown (E_dis, E_fc, E_pen, E_spen, E_joints, E_prior)
- Final energy in similar range (depending on iteration count)
- Results saved to `./objects/<object>/grasp_predictions/<hand>/<n>_contacts/graspqp/default/<object>.dexgrasp.pt`

Example output from sanity check (500 iterations):
```
Step 1: total=391.324 | E_dis=1.130, E_fc=47.786, ...
Step 500: total=50.123 | E_dis=0.150, E_fc=0.123, ...

Energy: best=25.12, mean=50.12, worst=120.45
==> Exported to /path/to/objects/apple/grasp_predictions/allegro/8_contacts/graspqp/default/apple.dexgrasp.pt
```

### Visualizing Results

After running either `fit.py` or `sanity_check_optim.py`, visualize the results:

```bash
# Visualize results (both fit.py and sanity_check_optim.py save to the same location)
python scripts/vis/visualize_result.py \
    --dataset ./objects \
    --num_assets 1 \
    --hand_name allegro \
    --num_contacts 8_contacts \
    --show \
    --max_grasps 32
```

This will:
1. Find the saved `.dexgrasp.pt` file in `./objects/apple/grasp_predictions/allegro/8_contacts/graspqp/default/`
2. Load and visualize up to 32 grasps
3. Save a rendered mesh to `./_vis/objects/apple/allegro/default/8_contacts/graspqp/render.glb`

**Note**: The `--num_contacts` should match the `--n_contact` used during optimization (default is 8, format is `8_contacts`).

### Viewing in Browser

The `--show` flag opens the visualization directly in your browser:

```bash
python scripts/vis/visualize_result.py \
    --dataset ./objects \
    --num_assets 1 \
    --hand_name allegro \
    --num_contacts 8_contacts \
    --show \
    --max_grasps 32
```

This will:
1. Load the grasp predictions
2. Open an interactive 3D viewer in your browser
3. Wait for you to press Enter in the terminal to continue

The visualization is also saved as a `.glb` file in `./_vis/` which you can open later with:
- Online viewer: https://gltf-viewer.donmccurdy.com/
- VS Code/Cursor: Install "glTF Tools" extension

## Module Structure

```
optim/
├── __init__.py           # Package exports
├── README.md             # This file
├── state.py              # TrajectoryState, ReferenceTrajectory
├── context.py            # OptimizationContext
├── problem.py            # OptimizationProblem
├── runner.py             # OptimizationRunner
├── costs/                # Cost functions
│   ├── base.py           # CostFunction ABC
│   ├── reference.py      # ReferenceTrackingCost
│   ├── penetration.py    # PenetrationCost, SelfPenetrationCost
│   ├── temporal.py       # VelocitySmoothnessCost, etc.
│   ├── grasp.py          # ContactDistanceCost, ForceClosureCost
│   └── registry.py       # Cost registration
├── optimizers/           # Optimizer implementations
│   ├── base.py           # Optimizer ABC
│   └── torch_optim.py    # Adam, SGD, LBFGS wrappers
└── callbacks/            # Optimization callbacks
    ├── base.py           # Callback ABC
    ├── logging.py        # LoggingCallback
    └── checkpoint.py     # CheckpointCallback
```

## Key Concepts

### Why MalaStar Works

The `MalaStar` optimizer (used by `fit.py`) properly handles PyTorch's computation graph by:

1. **Breaking graph history**: `set_parameters()` uses `.clone()` to create fresh tensors
2. **Correct loop order**: `try_step()` → `zero_grad()` → `calculate_energy()` → `backward()` → `accept_step()`
3. **Warm-start handling**: The QP solver caches solutions, but MalaStar's cloning prevents graph retention issues

### Common Issues

If you see `RuntimeError: Trying to backward through the graph a second time`:
- The optimizer is not properly breaking the computation graph between iterations
- Tensors from previous iterations are being reused without `.detach().clone()`
- The QP solver's warm-start feature may be holding references to old tensors

## Profiling & Testing

### Profiling Optimization Performance

Profile the optimization loop to identify bottlenecks:

```bash
python scripts/sanity_check_optim.py \
    --data_root_path ./objects \
    --object_code_list apple \
    --hand_name allegro \
    --batch_size 16 \
    --n_iter 500 \
    --prior_file configs/extracted_prior.yaml \
    --w_prior 100 \
    --profile
```

This outputs a detailed timing breakdown:
```
Section                               Total(s)   Mean(ms)       %
step                                    60.02     120.04   100.0%
energy                                  30.73      61.45    51.2%
  └─ qp_force_closure                   15.21      30.41    25.3%
  └─ sdf_contact                        12.09      24.18    20.1%
backward                                15.60      31.20    26.0%
try_step                                 7.92      15.85    13.2%
accept_step                              5.73      11.45     9.5%
```

### System Tests

Run the system test to validate correctness and detect performance regressions:

```bash
# Default: verbose mode (streams output, shows detailed comparisons)
python -m unittest tests.system.test_sanity_check -v

# Quiet mode (captures output)
VERBOSE=0 python -m unittest tests.system.test_sanity_check -v
```

The test checks:
- **Quality bounds**: E_dis, E_fc, E_pen, E_spen, E_prior within thresholds
- **Regression detection**: Results within 10% of baseline
- **Performance**: Timing within 20% of baseline

### Updating the Baseline

After making intentional changes that affect optimization results:

```bash
python scripts/update_baseline.py
```

This regenerates `tests/baselines/sanity_check_baseline.json` with the new expected values.

## Running Unit Tests

```bash
# Run all optim tests
python -m unittest discover -s tests/optim -v

# Run specific test file
python -m unittest tests/optim/test_state.py -v
python -m unittest tests/optim/test_costs.py -v
python -m unittest tests/optim/test_optimizers.py -v
```
