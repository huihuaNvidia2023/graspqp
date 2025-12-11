# GraspQP: Hierarchical Contact Sampling & Prior-Based Initialization

This document covers the new features for improved grasp optimization:
1. **Hierarchical Contact Sampling** - Efficient, configurable contact point selection
2. **Prior-Based Initialization** - Guide optimization with initial pose/contact hints

All configuration is done via a **single YAML file** passed with `--prior_file`.

---

## Quick Start

### Running Tests

```bash
# Activate environment
conda activate graspqp

# Run unit tests
python -m unittest graspqp/tests/test_contact_sampling.py -v

# Run interactive test with visualization
python scripts/test_contact_sampling.py --hand_name allegro --mode constrained --show_plotly
```

### Running fit.py

```bash
# Basic usage (no config file - uses default uniform sampling)
python scripts/fit.py \
    --data_root_path ./objects \
    --object_code_list apple \
    --hand_name allegro \
    --batch_size 32

# With config file (recommended)
python scripts/fit.py \
    --data_root_path ./objects \
    --object_code_list apple \
    --hand_name allegro \
    --batch_size 32 \
    --prior_file configs/fingertip_only.yaml
```

### Visualizing Results

```bash
# Visualize generated grasps with Plotly
python scripts/vis/visualize_result.py \
    --dataset ./objects \
    --num_assets 1 \
    --hand_name allegro \
    --show
```

---

## Example Configurations

### 1. Fingertip-Only Contacts (`configs/fingertip_only.yaml`)

```yaml
contact:
  mode: constrained
  links:
    - index_link_3
    - middle_link_3
    - ring_link_3
    - thumb_link_3
  min_fingers: 2
  max_contacts_per_link: 4

priors: []
```

### 2. Guided Sampling (`configs/guided_fingertips.yaml`)

```yaml
contact:
  mode: guided
  links:
    - index_link_3
    - middle_link_3
    - thumb_link_3
  preference_weight: 0.8  # 80% from preferred links
  min_fingers: 2

priors: []
```

### 3. With Prior Pose (`configs/with_prior_pose.yaml`)

```yaml
contact:
  mode: constrained
  links:
    - index_link_3
    - middle_link_3
    - thumb_link_3

priors:
  - translation: [0.0, 0.0, 0.1]
    rotation: [1, 0, 0, 0]  # quaternion (w,x,y,z)

jitter_translation: 0.02
jitter_rotation: 0.1
prior_weight: 10.0
```

### 4. Multiple Priors (`configs/multi_prior.yaml`)

```yaml
contact:
  mode: constrained
  links: [index_link_3, middle_link_3, ring_link_3, thumb_link_3]

priors:
  - translation: [0.0, 0.0, 0.12]
    rotation: [1, 0, 0, 0]
  - translation: [0.0, 0.1, 0.05]
    rotation: [0.707, 0.707, 0, 0]
  - translation: [0.05, 0.05, 0.1]
    rotation: [0.924, 0.383, 0, 0]

prior_weight: 8.0
```

### 5. Per-Batch Contacts (`configs/per_batch_contacts.yaml`)

```yaml
priors:
  # Pinch grasp
  - translation: [0.0, 0.0, 0.08]
    contact_links: [index_link_3, thumb_link_3]
    contact_mode: constrained

  # Three-finger
  - translation: [0.0, 0.0, 0.1]
    contact_links: [index_link_3, middle_link_3, thumb_link_3]
    contact_mode: constrained
```

---

## Configuration Reference

### Full YAML Schema

```yaml
# ============================================================
# Contact Sampling Configuration (global default)
# ============================================================
contact:
  mode: uniform          # "uniform", "guided", "constrained"
  links:                 # Link names for guided/constrained
    - index_link_3
    - thumb_link_3
  preference_weight: 0.8 # For guided: fraction from preferred links
  min_fingers: 2         # Minimum finger diversity
  max_contacts_per_link: 4

# ============================================================
# Prior Poses (optional)
# ============================================================
priors:
  - translation: [x, y, z]     # Position relative to object CoG
    rotation: [w, x, y, z]     # Quaternion (w,x,y,z format)
    joints:                    # Optional: specific joint angles
      index_joint_0: 0.3
      thumb_joint_0: 0.5
    contact_links: [...]       # Optional: override global contact.links
    contact_mode: constrained  # Optional: override global contact.mode
    contact_preference_weight: 0.8

# ============================================================
# Jitter for Batch Diversity
# ============================================================
jitter_translation: 0.02  # meters (std dev)
jitter_rotation: 0.1      # radians (std dev, ~6 degrees)
jitter_joints: 0.1        # radians (std dev)

# ============================================================
# Optimization Weight
# ============================================================
prior_weight: 10.0  # Weight for prior deviation energy
```

---

## Contact Sampling Modes

| Mode | Description |
|------|-------------|
| `uniform` | Random sampling from all contact candidates (default) |
| `guided` | Prefer specified links but allow others |
| `constrained` | Sample ONLY from specified links |

### Available Links (Allegro Hand)

| Finger | Link 1 | Link 2 | Link 3 (tip) |
|--------|--------|--------|--------------|
| Index  | `index_link_1` | `index_link_2` | `index_link_3` |
| Middle | `middle_link_1` | `middle_link_2` | `middle_link_3` |
| Ring   | `ring_link_1` | `ring_link_2` | `ring_link_3` |
| Thumb  | `thumb_link_1` | `thumb_link_2` | `thumb_link_3` |

---

## API Reference

### Python Usage

```python
from graspqp.core import (
    GraspPriorLoader,
    ContactSamplingConfig,
    HierarchicalContactSampler,
)

# Load config
config = GraspPriorLoader.load_from_file("configs/fingertip_only.yaml")

# Create contact sampler from config
contact_config = ContactSamplingConfig(
    mode=config.contact.mode,
    preferred_links=config.contact.links,
    min_fingers=config.contact.min_fingers,
)
sampler = HierarchicalContactSampler(hand_model, contact_config)

# Sample contacts
indices = sampler.sample(batch_size=32, n_contacts=12)

# Expand priors for initialization
if config.priors:
    prior_data = GraspPriorLoader.expand_priors(config, batch_size, hand_model, device)
    prior_pose = GraspPriorLoader.create_hand_pose_from_priors(prior_data)
```

---

## Tips

1. **Start with `configs/fingertip_only.yaml`** - constrains search space significantly
2. **Use priors when you have prior knowledge** - speeds up convergence
3. **Set `min_fingers: 2`** - ensures stable grasps
4. **Keep `prior_weight` around 5-15** - too high restricts exploration

---

## Troubleshooting

### "Cannot sample N contacts from M candidates"
- Add more links to `contact.links`
- Increase `max_contacts_per_link`
- Reduce `n_contacts` in fit.py

### Prior not affecting optimization
- Ensure `prior_weight > 0` in config
- Check priors list is not empty
- Verify translation/rotation format

### ROS2 import errors with pytest
Use unittest instead:
```bash
python -m unittest graspqp/tests/test_contact_sampling.py -v
```
