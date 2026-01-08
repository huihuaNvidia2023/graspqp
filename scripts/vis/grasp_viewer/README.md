# Grasp Trajectory Visualizer

Interactive visualization tool for grasp trajectories using [viser](https://github.com/nerfstudio-project/viser).

## Features

- **Multiple Trajectory Support**: Load and compare multiple optimized trajectories from a single checkpoint
- **Reference Overlay**: Overlay reference trajectory with adjustable opacity for comparison
- **View Modes**:
  - **Playback**: Animate through frames with play/pause controls
  - **Ghost**: View all frames simultaneously with opacity gradient
- **Interactive Controls**: Dropdown for trajectory selection, frame scrubber, speed control
- **Modular Design**: Easily extensible renderer architecture

## Installation

```bash
# Make sure viser is installed
pip install viser>=0.2.0

# Install the package
cd /path/to/graspqp
pip install -e graspqp/
```

## Usage

### Basic Usage

```bash
# View optimized trajectories
python -m scripts.vis.grasp_viewer.main --optimized path/to/optimized.dexgrasp.pt

# View with reference comparison
python -m scripts.vis.grasp_viewer.main \
    --optimized path/to/optimized.dexgrasp.pt \
    --reference path/to/reference.yaml

# View reference only
python -m scripts.vis.grasp_viewer.main --reference path/to/reference.yaml

# Specify port
python -m scripts.vis.grasp_viewer.main --optimized data.pt --port 8888

# Start in ghost view mode
python -m scripts.vis.grasp_viewer.main --optimized data.pt --view-mode ghost
```

### CLI Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--optimized` | `-o` | Path to optimized trajectory file (.dexgrasp.pt) |
| `--reference` | `-r` | Path to reference trajectory file (.yaml) |
| `--asset-dir` | | Path to graspqp assets directory |
| `--port` | `-p` | Port for viser server (default: 8080) |
| `--view-mode` | | Initial view mode: "playback" or "ghost" |

## Data Formats

### Optimized Trajectory (.dexgrasp.pt)

Standard graspqp checkpoint format. The viewer extracts trajectories from the batch dimension.

```python
{
    "trajectory": {
        "n_frames": int,
        "dt": float,
        "hand_states": Tensor,  # (B, T, D)
        "object_states": Tensor,  # (B, T, 7) [x,y,z, qw,qx,qy,qz]
    },
    "values": Tensor,  # (B,) energies
    "metadata": {
        "hand_name": str,
        "object_code": str,
        ...
    },
}
```

### Reference Trajectory (.yaml)

```yaml
metadata:
  name: "trajectory_name"
  n_frames: 100
  dt: 0.01

object:
  mesh_path: "/path/to/mesh.obj"
  scale: 1.0
  poses: <base64-encoded numpy array or path to .npy>

hands:
  - name: "right_hand"
    type: "allegro"
    urdf_path: "/path/to/hand.urdf"
    mesh_dir: "/path/to/meshes"
    qpos: <base64-encoded numpy array>  # (T, n_joints)
    # OR
    hand_states: <base64-encoded>  # (T, D) graspqp format
```

## Architecture

```
grasp_viewer/
├── data/
│   ├── types.py              # Data structures
│   ├── reference_adapter.py  # YAML loading
│   ├── optimized_adapter.py  # .dexgrasp.pt loading
│   └── loader.py             # Unified interface
├── renderers/
│   ├── base.py               # Abstract renderer
│   ├── hand_renderer.py      # URDF-based hand rendering
│   ├── object_renderer.py    # Object mesh rendering
│   ├── contact_renderer.py   # Contact points
│   └── label_renderer.py     # Frame labels
├── scene/
│   └── scene_manager.py      # Coordinates renderers
├── ui/
│   ├── control_panel.py      # GUI controls
│   ├── playback.py           # Animation controller
│   └── app.py                # Main application
└── main.py                   # CLI entry point
```

## Creating Test Data

```bash
# Create a test reference trajectory
python -m scripts.vis.grasp_viewer.examples.create_test_reference

# View it
python -m scripts.vis.grasp_viewer.main \
    --reference scripts/vis/grasp_viewer/examples/test_data/test_reference.yaml
```

## Extending

### Adding a New Renderer

1. Create a new file in `renderers/`
2. Inherit from `BaseRenderer`
3. Implement `update()` and `remove()` methods
4. Register in `scene_manager.py`

### Adding New Hand Types

Add the hand configuration to `HAND_CONFIGS` in `data/optimized_adapter.py`:

```python
HAND_CONFIGS = {
    "my_hand": {
        "urdf_file": "my_hand/robot.urdf",
        "mesh_dir": "my_hand/meshes",
    },
    ...
}
```
