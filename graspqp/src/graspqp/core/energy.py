"""Energy calculation with optional profiling support."""

import torch

from graspqp.utils.profiler import StepProfiler, get_profiler

# Default disabled profiler for when none is passed
_default_profiler = get_profiler(enabled=False)


def calculate_energy(
    hand_model,
    object_model,
    energy_fnc: any = None,
    energy_names=[],
    method="gendexgrasp",
    svd_gain=0.1,
    profiler: StepProfiler = None,
):
    """
    Calculate grasp energy terms.

    Args:
        hand_model: HandModel instance
        object_model: ObjectModel instance
        energy_fnc: Force closure energy function
        energy_names: List of additional energy terms to compute
        method: Distance computation method ("dexgraspnet" or "gendexgrasp")
        svd_gain: SVD regularization gain
        profiler: Optional StepProfiler for detailed timing

    Returns:
        Dict of energy terms
    """
    if profiler is None:
        profiler = _default_profiler

    batch_size, n_contact, _ = hand_model.contact_points.shape
    device = object_model.device

    losses = {}

    # E_dis: Contact distance energy (includes SDF computation)
    with profiler.section("energy.sdf_contact"):
        if method == "dexgraspnet":
            distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
            E_dis = torch.sum(distance.abs(), dim=-1, dtype=torch.float).to(device)
            losses["E_dis"] = E_dis
        elif method == "gendexgrasp":
            distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
            vC = contact_normal
            nH = hand_model.contact_normals
            E_dis = ((1 - torch.sum((-vC) * nH, dim=-1)).exp() * distance.abs()).sum(-1)
        else:
            raise ValueError(f"Unknown method: {method}")
        losses["E_dis"] = E_dis

    # E_fc: Force closure energy (QP solver)
    with profiler.section("energy.qp_force_closure"):
        E_fc, _lambda = energy_fnc(
            contact_pts=hand_model.contact_points,
            contact_normals=contact_normal,
            sdf=distance,
            cog=object_model.cog,
            with_solution=True,
            svd_gain=svd_gain,
        )
        losses["E_fc"] = E_fc

    # E_joints: Joint limit energy (fast, no profiling needed)
    E_joints = torch.sum(
        (hand_model.hand_pose[:, 9:] > hand_model.joints_upper)
        * (hand_model.hand_pose[:, 9:] - hand_model.joints_upper),
        dim=-1,
    ) + torch.sum(
        (hand_model.hand_pose[:, 9:] < hand_model.joints_lower)
        * (hand_model.joints_lower - hand_model.hand_pose[:, 9:]),
        dim=-1,
    )
    losses["E_joints"] = E_joints

    # E_pen: Penetration energy (SDF on surface points)
    with profiler.section("energy.sdf_penetration"):
        object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
        object_surface_points = (
            object_model.surface_points_tensor * object_scale
        )  # (n_objects * batch_size_each, num_samples, 3)
        distances = hand_model.cal_distance(object_surface_points)
        distances[distances <= 0] = 0
        E_pen = distances.sum(-1)
        losses["E_pen"] = E_pen

    # E_spen: Self-penetration energy
    with profiler.section("energy.self_penetration"):
        E_spen = hand_model.self_penetration()
        losses["E_spen"] = E_spen

    if "E_prior" in energy_names:
        forward_axis = (hand_model.global_rotation @ hand_model.grasp_axis.view(1, -1, 1)).view(-1, 3)
        # This should point downwards,
        axis_prior = torch.tensor([0, 0, -1], dtype=torch.float, device=device).view(1, 3)
        E_prior = 1 - torch.sum((forward_axis * axis_prior), dim=-1)
        losses["E_prior"] = E_prior

    if "E_wall" in energy_names:
        z_height = hand_model.get_surface_points()[..., -1].clamp(max=0.0)
        losses["E_wall"] = z_height.abs().sum(-1)

    if "E_manipulativity" in energy_names:
        with profiler.section("energy.manipulability"):
            E_jacobian = hand_model.get_manipulability(
                contact_normal * distance.unsqueeze(-1).abs().clamp(min=5e-3),
                hand_model.contact_point_indices,
            )
            E_manipulativity = E_jacobian.mean(-1)
            losses["E_manipulativity"] = E_manipulativity

    return losses
