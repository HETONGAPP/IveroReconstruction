# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/make_fragments.py

import math
import os, sys
import numpy as np
import open3d as o3d

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)

from open3d_example import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimize_posegraph import optimize_posegraph_for_fragment

# check opencv python package
with_opencv = initialize_opencv()
if with_opencv:
    from opencv_pose_estimation import pose_estimation

class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(folder):
    traj = []
    traj_file = os.path.join(folder, 'trajectory.log')
    with open(traj_file, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj


def pose_between_frame_id(source_id, target_id, extrinsics):
    source_to_world = extrinsics[source_id].pose
    world_to_target = np.linalg.inv(extrinsics[target_id].pose)
    source_to_target = np.matmul(world_to_target, source_to_world)
    
    return source_to_target

def register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic,
                           with_opencv, extrinsics, config):
    source_rgbd_image = read_rgbd_image(color_files[s], depth_files[s], True,
                                        config)
    target_rgbd_image = read_rgbd_image(color_files[t], depth_files[t], True,
                                        config)

    # source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, intrinsic)
    # source = source_pcd
    # source = source_pcd.voxel_down_sample(voxel_size=config["voxel_size"])
    # source.estimate_normals()

    # target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd_image, intrinsic)
    # target = target_pcd
    # target = target_pcd.voxel_down_sample(voxel_size=config["voxel_size"])
    # target.estimate_normals()
    
    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = config["depth_diff_max"]
    if abs(s - t) != 1:
        # if with_opencv:
        #     success_5pt, odo_init = pose_estimation(source_rgbd_image,
        #                                             target_rgbd_image,
        #                                             intrinsic, False)
        #     if success_5pt:
        #         [success, trans, info
        #         ] = o3d.pipelines.odometry.compute_rgbd_odometry(
        #             source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
        #             o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
        #             option)
        #         return [success, trans, info]
        
        
        # success=False
        # try:
        #     icp_fine = o3d.pipelines.registration.registration_colored_icp(
        #         source, target, 1,
        #         pose_between_frame_id(s, t, extrinsics), estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=0.78000), criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-07, relative_rmse=1.000000e-07, max_iteration=100))
        #     success=True
        #     trans = icp_fine.transformation
        # except:
        #     return [False, np.identity(4), np.identity(6)]
        # success=True
        # trans = pose_between_frame_id(s, t, extrinsics)
        # info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        #         source, target, 1,
        #         trans)
        # return [success, trans, info]
    
        return [False, np.identity(4), np.identity(6)]
    else:
        [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, intrinsic, pose_between_frame_id(s, t, extrinsics),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        
        # success=False
        # try:
        #     icp_fine = o3d.pipelines.registration.registration_colored_icp(
        #         source, target, 0.07,
        #         pose_between_frame_id(s, t, extrinsics), estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=0.78000), criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-07, relative_rmse=1.000000e-07, max_iteration=100))
        #     success=True
        #     trans = icp_fine.transformation
        # except:
        #     return [False, np.identity(4), np.identity(6)]
        # success=True
        # trans = pose_between_frame_id(s, t, extrinsics)
        # info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        #         source, target, 0.07,
        #         trans)
        return [success, trans, info]


def make_posegraph_for_fragment(path_dataset, sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, extrinsics, config):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    trans_odometry = np.identity(4)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(trans_odometry))
    for s in range(sid, eid):
        for t in range(s + 1, eid):
            # odometry
            if t == s + 1:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic, with_opencv, extrinsics, config)
                trans_odometry = np.dot(trans, trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        trans_odometry_inv))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(s - sid,
                                                             t - sid,
                                                             trans,
                                                             info,
                                                             uncertain=False))

            # keyframe loop closure
            if s % config['n_keyframes_per_n_frame'] == 0 \
                    and t % config['n_keyframes_per_n_frame'] == 0:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic, with_opencv, extrinsics, config)
                if success:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            s - sid, t - sid, trans, info, uncertain=True))
    o3d.io.write_pose_graph(
        join(path_dataset, config["template_fragment_posegraph"] % fragment_id),
        pose_graph)


def integrate_rgb_frames_for_fragment(color_files, depth_files, fragment_id,
                                      n_fragments, pose_graph_name, intrinsic,
                                      config):
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i in range(len(pose_graph.nodes)):
        i_abs = fragment_id * config['n_frames_per_fragment'] + i
        print(
            "Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." %
            (fragment_id, n_fragments - 1, i_abs, i + 1, len(pose_graph.nodes)))
        rgbd = read_rgbd_image(color_files[i_abs], depth_files[i_abs], False,
                               config)
        pose = pose_graph.nodes[i].pose
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def make_pointcloud_for_fragment(path_dataset, color_files, depth_files,
                                 fragment_id, n_fragments, intrinsic, config):
    mesh = integrate_rgb_frames_for_fragment(
        color_files, depth_files, fragment_id, n_fragments,
        join(path_dataset,
             config["template_fragment_posegraph_optimized"] % fragment_id),
        intrinsic, config)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd_name = join(path_dataset,
                    config["template_fragment_pointcloud"] % fragment_id)
    o3d.io.write_point_cloud(pcd_name, pcd, False, True)


def process_single_fragment(fragment_id, color_files, depth_files, n_files,
                            n_fragments, extrinsics, config):
    if config["path_intrinsic"]:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(
            config["path_intrinsic"])
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    sid = fragment_id * config['n_frames_per_fragment']
    eid = min(sid + config['n_frames_per_fragment'] + 20, n_files)

    make_posegraph_for_fragment(config["path_dataset"], sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, extrinsics, config)
    optimize_posegraph_for_fragment(config["path_dataset"], fragment_id, config)
    make_pointcloud_for_fragment(config["path_dataset"], color_files,
                                 depth_files, fragment_id, n_fragments,
                                 intrinsic, config)


def run(config):

    print("making fragments from RGBD sequence.")
    make_clean_folder(join(config["path_dataset"], config["folder_fragment"]))
    extrinsics = read_trajectory(config["path_dataset"])

    [color_files, depth_files] = get_rgbd_file_lists(config["path_dataset"])
    n_files = len(color_files)
    n_fragments = int(
        math.ceil(float(n_files) / config['n_frames_per_fragment']))

    if config["python_multi_threading"] is True:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = 8
        Parallel(n_jobs=MAX_THREAD)(delayed(process_single_fragment)(
            fragment_id, color_files, depth_files, n_files, n_fragments, extrinsics, config)
                                    for fragment_id in range(n_fragments))
    else:
        for fragment_id in range(n_fragments):
            process_single_fragment(fragment_id, color_files, depth_files,
                                    n_files, n_fragments, extrinsics, config)