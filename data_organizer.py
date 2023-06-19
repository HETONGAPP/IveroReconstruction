from scipy.spatial.transform import Rotation as R
import argparse
import numpy as np
import os
import copy
import shutil
import cv2
import json

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Dataset Converter')
    parser.add_argument('--path', required=True, help="Kindly provide path for the data directory")

    args = parser.parse_args()

    tum_trajectory=os.path.join(args.path, 'poses.txt')
    trajectory_log=os.path.join(args.path, 'trajectory.log')
    world_converter = np.empty([4, 4])

    with open(tum_trajectory, "r") as read_f:    
        with open(trajectory_log,'wb') as write_f:
            for count, line in enumerate(read_f):
                line_items = line.rstrip().split(' ')
                if line_items[0]=='#':
                    continue
                else:
                    pose_mat = np.identity(4)

                    metadata = [[count, count, count+1]]
                    # metadata = [[count, count, count+1]]
                    pose_mat[:3, :3] = R.from_quat(line_items[4:8]).as_matrix()
                    pose_mat[:3, 3] = line_items[1:4]
                    if count==0:
                        world_converter = copy.deepcopy(pose_mat)
                        world_converter[:3, :3] = pose_mat[:3, :3].transpose()
                        trans_vec = pose_mat[:3, 3]
                        world_converter[:3, 3] = -1*np.matmul(pose_mat[:3, :3].transpose(), trans_vec[..., None])[:3, 0]
                        print(world_converter)
                    
                    pose_mat = np.matmul(world_converter, pose_mat)
                    np.savetxt(write_f, metadata, delimiter=' ', fmt='%i')
                    for line in pose_mat:
                        np.savetxt(write_f, [line], delimiter=' ')
                    
                    old_depth = os.path.join(args.path, 'depth', "{:.6f}".format(float(line_items[0]))+'.png')
                    old_rgb = os.path.join(args.path, 'rgb', "{:.6f}".format(float(line_items[0]))+'.png')

                    new_depth = os.path.join(args.path, 'depth_tmp', str(count+1)+'.png')
                    new_rgb = os.path.join(args.path, 'rgb_tmp', str(count+1)+'.png')

                    if not os.path.isdir(os.path.join(args.path, 'depth_tmp')):
                        os.makedirs(os.path.join(args.path, 'depth_tmp'))

                    if not os.path.isdir(os.path.join(args.path, 'rgb_tmp')):
                        os.makedirs(os.path.join(args.path, 'rgb_tmp'))

                    os.rename(old_depth, new_depth)
                    os.rename(old_rgb, new_rgb)

    for filename in os.listdir(os.path.join(args.path, 'depth')):
        file_path = os.path.join(os.path.join(args.path, 'depth'), filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    for filename in os.listdir(os.path.join(args.path, 'rgb')):
        file_path = os.path.join(os.path.join(args.path, 'rgb'), filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    
    from_dir = os.path.join(args.path, 'depth_tmp')
    to_dir = os.path.join(args.path, 'depth')

    for file in os.listdir(from_dir):
        shutil.move(os.path.join(from_dir, file), to_dir)
    os.rmdir(from_dir)

    from_dir = os.path.join(args.path, 'rgb_tmp')
    to_dir = os.path.join(args.path, 'rgb')

    for file in os.listdir(from_dir):
        shutil.move(os.path.join(from_dir, file), to_dir)
    os.rmdir(from_dir)

    img = cv2.imread(os.path.join(args.path, 'rgb', '1.png'))

    if img.shape[1] == 640:
        shutil.copy(os.path.join(os.path.realpath(os.path.dirname(__file__)), 'calibration_640.json'), os.path.join(args.path, 'calibration.json'))
    else:
        shutil.copy(os.path.join(os.path.realpath(os.path.dirname(__file__)), 'calibration_848.json'), os.path.join(args.path, 'calibration.json'))

    json_filename = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'ivero.json')
    with open(json_filename) as fp:
        listObj = json.load(fp)

    listObj.update({
        "path_dataset": args.path,
        "path_intrinsic": os.path.join(args.path, 'calibration.json')
    })

    print(listObj)
    
    out_json_filename = os.path.join(args.path, 'ivero.json')

    with open(out_json_filename, 'w') as json_file:
        json.dump(listObj, json_file, 
                            indent=4,  
                            separators=(',',': '))