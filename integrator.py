# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Align multiple pieces of geometry in a global space"""

import open3d as o3d
import numpy as np

extrinsics = None

class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
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

def load_point_clouds(voxel_size=0.0):
    pcd_data = o3d.data.DemoICPPointClouds()
    pcds = []
    for i in range(3):
        pcd = o3d.io.read_point_cloud(pcd_data.paths[i])
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds


def pairwise_registration(source, target, source_id, target_id, max_correspondence_distance_coarse,
                          max_correspondence_distance_fine):
    print("Apply point-to-plane ICP")
    # icp_coarse = o3d.pipelines.registration.registration_icp(
    #     source, target, max_correspondence_distance_coarse, pose_between_frame_id(source_id, target_id),
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_generalized_icp(
        source, target, max_correspondence_distance_fine,
        pose_between_frame_id(source_id, target_id))
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], source_id, target_id,
                max_correspondence_distance_coarse,
                max_correspondence_distance_fine)
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

def pose_between_frame_id(source_id, target_id):
    source_to_world = extrinsics[source_id].pose
    world_to_target = np.linalg.inv(extrinsics[target_id].pose)
    source_to_target = np.matmul(world_to_target, source_to_world)
    
    return source_to_target

if __name__ == "__main__":
    # Frame 1
    color_raw = o3d.io.read_image('/home/jpanda001/Workplace/Data/orb_data3/rgb/1.png')
    depth_raw = o3d.io.read_image('/home/jpanda001/Workplace/Data/orb_data3/depth/1.png')
    rgbd_image_1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)

    # Frame 2
    color_raw = o3d.io.read_image('/home/jpanda001/Workplace/Data/orb_data3/rgb/30.png')
    depth_raw = o3d.io.read_image('/home/jpanda001/Workplace/Data/orb_data3/depth/30.png')
    rgbd_image_2 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)
    
    print(rgbd_image_2)
    extrinsics = read_trajectory('/home/jpanda001/Workplace/Data/orb_data3/trajectory.log')
    camera_intrinsics = o3d.io.read_pinhole_camera_intrinsic('/home/jpanda001/Workplace/Data/orb_data3/calibration.json')
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_1, camera_intrinsics)
    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_2, camera_intrinsics)
    print('--------------------')
    voxel_size = 0.002
    pcds_down = []
    pcd1_down = pcd1.voxel_down_sample(voxel_size=voxel_size)
    pcd1_down.estimate_normals()
    pcds_down.append(pcd1_down)
    pcd2_down = pcd2.voxel_down_sample(voxel_size=voxel_size)
    pcd2_down.estimate_normals()
    pcds_down.append(pcd2_down)
    # o3d.visualization.draw([pcd])

    print("Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds_down,
                                       max_correspondence_distance_coarse,
                                       max_correspondence_distance_fine, tf)

    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

    print("Transform points and display")
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    o3d.visualization.draw(pcds_down)