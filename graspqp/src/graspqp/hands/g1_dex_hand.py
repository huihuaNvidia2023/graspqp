import json
import os

import torch

from graspqp.core import HandModel


def getHandModel(device: str, asset_dir: str, grasp_type: str = "all", **kwargs) -> HandModel:
    contact_links = None

    if grasp_type is not None and grasp_type != "all":
        eigengrasp_file = f"{asset_dir}/g1_dex_hand/eigengrasps.json"
        if not os.path.exists(eigengrasp_file):
            raise ValueError(f"eigengrasps.json not found at {eigengrasp_file}")
        json_data = json.load(open(eigengrasp_file))
        if grasp_type not in json_data:
            raise ValueError(
                f"grasp type {grasp_type} not found in eigengrasps.json. Available grasp types are {list(json_data.keys())}"
            )
        contact_links = json_data[grasp_type]

    params = dict(
        mjcf_path=f"{asset_dir}/g1_dex_hand/g1_right_hand.urdf",
        mesh_path=f"{asset_dir}/g1_dex_hand/meshes",
        contact_points_path=f"{asset_dir}/g1_dex_hand/contact_points.json",
        penetration_points_path=f"{asset_dir}/g1_dex_hand/penetration_points.json",
        contact_links=contact_links,
        device=device,
        n_surface_points=512,
        # G1 hand geometry (after URDF mount rotation):
        # - Fingers extend in +X direction
        # - Palm normal faces +Z direction (same as Allegro)
        # 
        # Match Allegro's axis convention:
        # - forward_axis="z": palm faces toward object
        # - up_axis="x": finger extension direction is "up" 
        # - grasp_axis="y": lateral direction
        forward_axis="z",
        up_axis="x",
        grasp_axis="y",
        use_collision_if_possible=True,
        only_use_collision=True,
        default_state=torch.tensor(
            [
                0.0,    # right_hand_thumb_0_joint - neutral spread
                -0.1,   # right_hand_thumb_1_joint - slightly flexed (near neutral)
                -0.4,   # right_hand_thumb_2_joint - slightly flexed (near neutral)
                0.47,   # right_hand_middle_0_joint - partially flexed (30% of range)
                0.53,   # right_hand_middle_1_joint - partially flexed (30% of range)
                0.47,   # right_hand_index_0_joint - partially flexed (30% of range)
                0.53,   # right_hand_index_1_joint - partially flexed (30% of range)
            ],
            dtype=torch.float,
            device=device,
        ),
        grasp_type=grasp_type,
        # Offset to bring palm closer to object
        # Format: [forward, up, left] in hand's local frame (forward=z, up=x, left=y)
        # Positive forward moves hand toward object
        init_offset=torch.tensor([-0.04, 0.0, -0.10], dtype=torch.float, device=device),
    )
    params.update(kwargs)
    return HandModel(**params)
