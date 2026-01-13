#!/usr/bin/env python
"""
Quick test script for iterating on G1 dex hand initial configuration.

Usage:
    python scripts/test_g1_init.py

Modify the g1_dex_hand.py parameters and re-run this script to see changes.
"""
import torch
import plotly.graph_objects as go
import plotly.io as pio
from graspqp.hands import get_hand_model
from graspqp.core import ObjectModel
from graspqp.core.initializations import initialize_convex_hull
import numpy as np
from argparse import Namespace

pio.renderers.default = 'browser'

def main():
    # Create models
    hand_model = get_hand_model('g1_dex_hand', 'cuda')
    object_model = ObjectModel('./objects', batch_size_each=1, device='cuda', scale=1.0, num_samples=1000)
    object_model.initialize(['apple'])

    # Initialization args
    args = Namespace(
        distance_lower=0.08,
        distance_upper=0.12,
        rotate_lower=-3.14,
        rotate_upper=3.14,
        pitch_lower=-0.5,
        pitch_upper=0.5,
        tilt_lower=-0.3,
        tilt_upper=0.3,
        jitter_strength=0.0,
    )

    # Use fixed seed for reproducibility
    torch.manual_seed(42)
    translation, rotation_6d, joint_angles = initialize_convex_hull(
        hand_model, object_model, args, init_contacts=False
    )
    hand_pose = torch.cat([translation, rotation_6d, joint_angles], dim=-1)
    hand_model.set_parameters(hand_pose, contact_point_indices='all')

    # Object mesh at origin
    obj_mesh = object_model.object_mesh_list[0]
    obj_verts = np.array(obj_mesh.vertices) * object_model.object_scale_tensor[0].cpu().numpy()
    obj_faces = np.array(obj_mesh.faces)

    # Visualize
    hand_data = hand_model.get_plotly_data(0, opacity=0.8, color='lightgray', with_contact_points=True)
    obj_plot = go.Mesh3d(
        x=obj_verts[:, 0], y=obj_verts[:, 1], z=obj_verts[:, 2],
        i=obj_faces[:, 0], j=obj_faces[:, 1], k=obj_faces[:, 2],
        color='lightgreen', opacity=0.7, name='apple'
    )

    # Coordinate axes at origin
    scale = 0.1
    x_axis = go.Scatter3d(x=[0, scale], y=[0, 0], z=[0, 0], mode='lines', 
                          line=dict(color='red', width=5), name='X')
    y_axis = go.Scatter3d(x=[0, 0], y=[0, scale], z=[0, 0], mode='lines', 
                          line=dict(color='green', width=5), name='Y')
    z_axis = go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, scale], mode='lines', 
                          line=dict(color='blue', width=5), name='Z')

    fig = go.Figure(data=hand_data + [obj_plot, x_axis, y_axis, z_axis])
    fig.update_layout(scene_aspectmode='data', title='G1 Dex Hand Initial Configuration')
    fig.show()

    # Print current configuration
    print(f"\n=== Current Configuration ===")
    print(f"Hand translation: {hand_model.global_translation[0].cpu().numpy()}")
    print(f"Joint angles: {joint_angles[0].cpu().numpy()}")
    print(f"Forward axis: {hand_model.forward_axis.cpu().numpy()}")
    print(f"Up axis: {hand_model.up_axis.cpu().numpy()}")
    print(f"Init offset: {hand_model.init_offset.cpu().numpy() if hand_model.init_offset is not None else 'None'}")


if __name__ == "__main__":
    main()
