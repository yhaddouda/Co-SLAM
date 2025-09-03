import argparse
import os
import random
import time

import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree

# Headless renderer (Open3D 0.18)
from open3d.visualization import rendering as o3dr

'''
reconstruction evaluation tools
modified from https://github.com/cvg/nice-slam/blob/master/src/tools/eval_recon.py
This version uses Open3D OffscreenRenderer to avoid any DISPLAY/GLFW requirements.
Compatible with Open3D 0.18 (no MaterialRecord.double_sided; set_projection uses 3x3 intrinsics).
'''


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp


def get_align_transformation(rec_meshfile, gt_meshfile):
    """
    Align reconstructed mesh to the ground truth via point-to-point ICP.
    """
    o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation


def check_proj(points, W, H, fx, fy, cx, cy, c2w):
    """
    Check if points can be projected into the camera view.
    """
    c2w = c2w.copy()
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0
    points = torch.from_numpy(points).cuda().clone()
    w2c = np.linalg.inv(c2w)
    w2c = torch.from_numpy(w2c).cuda().float()
    K = torch.from_numpy(
        np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).cuda()
    ones = torch.ones_like(points[:, 0]).reshape(-1, 1).cuda()
    homo_points = torch.cat([points, ones], dim=1).reshape(-1, 4, 1).cuda().float()
    cam_cord_homo = w2c @ homo_points
    cam_cord = cam_cord_homo[:, :3]
    cam_cord[:, 0] *= -1
    uv = K.float() @ cam_cord.float()
    z = uv[:, -1:] + 1e-5
    uv = uv[:, :2] / z
    uv = uv.float().squeeze(-1).cpu().numpy()
    edge = 0
    mask = (0 <= -z[:, 0, 0].cpu().numpy()) & (uv[:, 0] < W - edge) & (uv[:, 0] > edge) & \
           (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
    return mask.sum() > 0


def calc_3d_mesh_metric(mesh_rec, mesh_gt, align=False):
    """
    3D reconstruction metric on already loaded meshes.
    """
    rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])

    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices)

    accuracy_rec *= 100
    completion_rec *= 100
    completion_ratio_rec *= 100

    return {'acc': accuracy_rec, 'comp': completion_rec, 'comp%': completion_ratio_rec}


def calc_3d_metric(rec_meshfile, gt_meshfile, align=True):
    """
    3D reconstruction metric from mesh files.
    """
    mesh_rec = trimesh.load(rec_meshfile, process=False)
    mesh_gt = trimesh.load(gt_meshfile, process=False)

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        mesh_rec = mesh_rec.apply_transform(transformation)

    rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])

    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices)

    accuracy_rec *= 100  # cm
    completion_rec *= 100  # cm
    completion_ratio_rec *= 100  # %

    print('accuracy: ', accuracy_rec)
    print('completion: ', completion_rec)
    print('completion ratio: ', completion_ratio_rec)

    return {'acc': accuracy_rec, 'comp': completion_rec, 'comp ratio': completion_ratio_rec}


def get_cam_position(gt_meshfile, sx=0.3, sy=0.6, sz=0.6, dx=0.0, dy=0.0, dz=0.0):
    mesh_gt = trimesh.load(gt_meshfile)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh_gt)
    extents[2] *= sz
    extents[1] *= sy
    extents[0] *= sx
    transform = np.linalg.inv(to_origin)
    transform[0, 3] += dx
    transform[1, 3] += dy
    transform[2, 3] += dz
    return extents, transform


def calc_2d_metric(rec_meshfile, gt_meshfile, unseen_gt_pcd_file,
                   pose_file=None, gt_depth_render_file=None,
                   depth_render_file=None, suffix="virt_cams", align=True,
                   n_imgs=1000, not_counting_missing_depth=True,
                   sx=0.3, sy=0.6, sz=0.6, dx=0.0, dy=0.0, dz=0.0):
    """
    2D reconstruction metric (Depth L1) using headless Open3D OffscreenRenderer.
    """
    H = 500
    W = 500
    focal = 300
    fx = focal
    fy = focal
    cx = H / 2.0 - 0.5
    cy = W / 2.0 - 0.5

    gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)

    # Ensure normals (helps renderer)
    if not gt_mesh.has_vertex_normals():
        gt_mesh.compute_vertex_normals()
    if not rec_mesh.has_vertex_normals():
        rec_mesh.compute_vertex_normals()

    pc_unseen = np.load(unseen_gt_pcd_file)

    if pose_file and os.path.exists(pose_file):
        sampled_poses = np.load(pose_file)["poses"]
        assert len(sampled_poses) == n_imgs
        print("Found saved renering poses! Loading from disk!!!")
    else:
        sampled_poses = None
        print("Saved renering poses NOT FOUND! Will do the sampling")
    if gt_depth_render_file and os.path.exists(gt_depth_render_file):
        gt_depth_renderings = np.load(gt_depth_render_file)["depths"]
        assert len(gt_depth_renderings) == n_imgs
        print("Found saved renered gt depths! Loading from disk!!!")
    else:
        gt_depth_renderings = None
        print("Saved renered gt depths NOT FOUND! Will re-render!!!")
    if depth_render_file and os.path.exists(depth_render_file):
        depth_renderings = np.load(depth_render_file)["depths"]
        assert len(depth_renderings) == n_imgs
        print("Found saved renered reconstructed depth! Loading from disk!!!")
    else:
        depth_renderings = None
        print("Saved renered reconstructed depth NOT FOUND! Will re-render!!!")

    gt_dir = os.path.dirname(unseen_gt_pcd_file)
    log_dir = os.path.dirname(rec_meshfile)

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        rec_mesh = rec_mesh.transform(transformation)

    # Vacant area inside the room for pose sampling
    extents, transform = get_cam_position(gt_meshfile, sx=sx, sy=sy, sz=sz, dx=dx, dy=dy, dz=dz)

    # -----------------------------
    # Offscreen renderer setup
    # -----------------------------
    renderer = o3dr.OffscreenRenderer(W, H)

    # Material (Open3D 0.18 — no double_sided attribute here)
    mat = o3dr.MaterialRecord()
    mat.shader = "defaultLit"

    scene = renderer.scene
    scene.set_background([0, 0, 0, 0])  # transparent bg

    def set_camera_from_c2w(c2w):
        # Open3D expects extrinsic as world-to-camera
        extr = np.linalg.inv(c2w).astype(np.float32)
        # Build intrinsics object (for setup) and also keep 3x3 for set_projection
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        renderer.setup_camera(intr, extr)
        # Correct signature for O3D 0.18:
        # set_projection(intrinsics: np.ndarray[3,3], near, far, image_width, image_height)
        scene.camera.set_projection(intr.intrinsic_matrix, 0.1, 20.0, float(W), float(H))

    def render_depth(mesh, name):
        # Replace geometry if already exists
        try:
            scene.remove_geometry(name)
        except Exception:
            pass
        scene.add_geometry(name, mesh, mat)
        depth_img = renderer.render_to_depth_image(z_in_view_space=True)
        return np.asarray(depth_img)

    errors = []
    poses = []
    gt_depths = []
    depths = []

    for i in range(n_imgs):
        if sampled_poses is None:
            while True:
                up = [0, 0, -1]
                origin = trimesh.sample.volume_rectangular(extents, 1, transform=transform).reshape(-1)
                tx = round(random.uniform(-10000, +10000), 2)
                ty = round(random.uniform(-10000, +10000), 2)
                tz = round(random.uniform(-10000, +10000), 2)
                target = np.array([tx, ty, tz]) - np.array(origin)
                c2w = viewmatrix(target, up, origin)
                tmp = np.eye(4)
                tmp[:3, :] = c2w
                c2w = tmp
                seen = check_proj(pc_unseen, W, H, fx, fy, cx, cy, c2w)
                if (~seen):
                    break
            poses.append(c2w)
        else:
            c2w = sampled_poses[i]

        set_camera_from_c2w(c2w)

        if gt_depth_renderings is None:
            gt_depth = render_depth(gt_mesh, "gt_mesh")
            gt_depths.append(gt_depth)
        else:
            gt_depth = gt_depth_renderings[i]

        if depth_renderings is None:
            ours_depth = render_depth(rec_mesh, "rec_mesh")
            depths.append(ours_depth)
        else:
            ours_depth = depth_renderings[i]

        if not_counting_missing_depth:
            valid_mask = (gt_depth > 0.) & (gt_depth < 19.)
            if np.count_nonzero(valid_mask) <= 100:
                continue
            errors.append(np.abs(gt_depth[valid_mask] - ours_depth[valid_mask]).mean())
        else:
            errors.append(np.abs(gt_depth - ours_depth).mean())

    # Save caches if they were not supplied
    if pose_file is None:
        np.savez(os.path.join(gt_dir, "sampled_poses_{}.npz".format(n_imgs)), poses=poses)
    elif not os.path.exists(pose_file):
        np.savez(pose_file, poses=poses)

    if gt_depth_render_file is None:
        np.savez(os.path.join(gt_dir, "gt_depths_{}.npz".format(n_imgs)), depths=gt_depths)
    elif not os.path.exists(gt_depth_render_file):
        np.savez(gt_depth_render_file, depths=gt_depths)

    if depth_render_file is None:
        np.savez(os.path.join(log_dir, "depths_{}_{}.npz".format(suffix, n_imgs)), depths=depths)
    elif not os.path.exists(depth_render_file):
        np.savez(depth_render_file, depths=depths)

    errors = np.array(errors)
    print('Depth L1: ', errors.mean() * 100)  # m -> cm
    return {"Depth L1": errors.mean() * 100}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments to evaluate the reconstruction.")
    parser.add_argument("--rec_mesh", type=str, help="reconstructed mesh file path")
    parser.add_argument("--gt_mesh", type=str, help="ground truth mesh file path")
    parser.add_argument("--dataset_type", type=str, default="Replica", help="dataset type: [Replica, RGBD]")
    parser.add_argument("-2d", "--metric_2d", action="store_true", help="enable 2D metric")
    parser.add_argument("-3d", "--metric_3d", action="store_true", help="enable 3D metric")
    args = parser.parse_args()

    if args.metric_3d:
        calc_3d_metric(args.rec_mesh, args.gt_mesh)

    if args.metric_2d:
        assert args.dataset_type in ["Replica", "RGBD"], "Unknown dataset type..."
        eval_data_dir = os.path.dirname(args.gt_mesh)
        unseen_pc_file = os.path.join(eval_data_dir, "gt_pc_unseen.npy")
        pose_file = os.path.join(eval_data_dir, "sampled_poses_1000.npz")
        if args.dataset_type == "Replica":  # follow NICE-SLAM
            sx, sy, sz, dx, dy, dz = 0.3, 0.7, 0.7, 0.0, 0.0, 0.4
        elif os.path.basename(eval_data_dir) == "complete_kitchen":
            sx, sy, sz, dx, dy, dz = 0.3, 0.5, 0.5, 1.2, 0.0, 1.8
        else:
            sx, sy, sz, dx, dy, dz = 0.3, 0.6, 0.6, 0.0, 0.0, 0.0
        calc_2d_metric(
            args.rec_mesh, args.gt_mesh, unseen_pc_file,
            pose_file=pose_file, n_imgs=1000, not_counting_missing_depth=True,
            sx=sx, sy=sy, sz=sz, dx=dx, dy=dy, dz=dz
        )

