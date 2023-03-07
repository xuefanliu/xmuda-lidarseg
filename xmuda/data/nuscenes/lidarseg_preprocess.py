import os
import os.path as osp

import numpy as np
import pickle

from nuscenes.eval.lidarseg.utils import LidarsegClassMapper
from nuscenes.nuscenes import NuScenes

from xmuda.data.nuscenes.lidarseg_nuscenes_dataloader import NuScenesBase
from xmuda.data.nuscenes.projection import map_pointcloud_to_image
from xmuda.data.nuscenes import splits

class_names_to_id = dict(zip(NuScenesBase.class_names, range(len(NuScenesBase.class_names))))
if 'background' in class_names_to_id:
    del class_names_to_id['background']


def preprocess(nusc, split_names, root_dir, out_dir,
               keyword=None, keyword_action=None, subset_name=None,
               location=None):
    # cannot process day/night and location at the same time
    assert not (bool(keyword) and bool(location))
    if keyword:
        assert keyword_action in ['filter', 'exclude']

    # init dict to save
    pkl_dict = {}
    for split_name in split_names:
         pkl_dict[split_name] = []

    for i, sample in enumerate(nusc.sample):
        curr_scene_name = nusc.get('scene', sample['scene_token'])['name']

        # get if the current scene is in train, val or test
        curr_split = None
        for split_name in split_names:
            if curr_scene_name in getattr(splits, split_name):
                curr_split = split_name
                break
        if curr_split is None:
            continue

        if subset_name == 'night':
            if curr_split == 'train':
                if curr_scene_name in splits.val_night:
                    curr_split = 'val'
        if subset_name == 'singapore':
            if curr_split == 'train':
                if curr_scene_name in splits.val_singapore:
                    curr_split = 'val'

        # filter for day/night
        if keyword:
            scene_description = nusc.get("scene", sample["scene_token"])["description"]
            if keyword.lower() in scene_description.lower():
                if keyword_action == 'exclude':
                    # skip sample
                    continue
            else:
                if keyword_action == 'filter':
                    # skip sample
                    continue

        if location:
            scene = nusc.get("scene", sample["scene_token"])
            if location not in nusc.get("log", scene['log_token'])['location']:
                continue

        lidar_token = sample["data"]["LIDAR_TOP"]
        cam_front_token = sample["data"]["CAM_FRONT"]
        lidar_path, boxes_lidar, _ = nusc.get_sample_data(lidar_token)
        cam_path, boxes_front_cam, cam_intrinsic = nusc.get_sample_data(cam_front_token)

        print('{}/{} {} {}'.format(i + 1, len(nusc.sample), curr_scene_name, lidar_path))

        sd_rec_lidar = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        cs_record_lidar = nusc.get('calibrated_sensor',
                             sd_rec_lidar['calibrated_sensor_token'])
        pose_record_lidar = nusc.get('ego_pose', sd_rec_lidar['ego_pose_token'])
        sd_rec_cam = nusc.get('sample_data', sample['data']["CAM_FRONT"])
        cs_record_cam = nusc.get('calibrated_sensor',
                             sd_rec_cam['calibrated_sensor_token'])
        pose_record_cam = nusc.get('ego_pose', sd_rec_cam['ego_pose_token'])

        calib_infos = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam['translation'],
            "ego2global_rotation_cam": pose_record_cam['rotation'],
            "cam2ego_translation": cs_record_cam['translation'],
            "cam2ego_rotation": cs_record_cam['rotation'],
            "cam_intrinsic": cam_intrinsic,
        }

        # load lidar points (3,N)
        pts = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :3].T

        #########################################################################################
        #    这部分是将点云投影到二维图像上，具体的过程在projection.py
        #########################################################################################
        # map point cloud into front camera image
        # pts_valid_flag: (N,)一个bool数组，表示pts_cam_coord中的每一个点是否有效
        # pts_cam_coord: (N,3)将点云转换到相机视角坐标系下点的坐标的集合（包含当前点云的所有点）,是三维的
        # pts_img: (N',2)将点云投影到二维平面后的有效点的集合，是二维的（如果点投影以后超出了图像范围则无效，如果点在相机背后，也视为无效）
        pts_valid_flag, pts_cam_coord, pts_img = map_pointcloud_to_image(pts, (900, 1600, 3), calib_infos)
        # fliplr so that indexing is row, col and not col, row ,只是交换了数组的两列，不是转置
        # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        pts_img = np.ascontiguousarray(np.fliplr(pts_img))

        # #########################################################################################
        # #    处理标签
        # #########################################################################################
        # 从文件中读取点云的标签
        fine_label_path = osp.join(root_dir,'lidarseg', nusc.version,lidar_token+'_lidarseg.bin')
        fine_labels = np.fromfile(fine_label_path, dtype=np.uint8)
        # 只保留有效点的坐标和label
        pts = pts[:, pts_valid_flag]  # (3,N')
        fine_labels = fine_labels[pts_valid_flag]
        # 此时的label是fine_label，将其转化为corse_label
        mapper = LidarsegClassMapper(nusc)
        num_pts = pts.shape[1]
        seg_labels=np.full(num_pts, fill_value=len(class_names_to_id), dtype=np.uint8) # 先全部填17
        for i in range(0,len(fine_labels)):
            seg_labels[i] = mapper.get_fine_idx_2_coarse_idx()[fine_labels[i]]
        # ↑这个地方的前提是能直接让seg_label等于上面那一串的前提是：
        #   在NuScenesBase定义class_names的时候的排列顺序与官方代码里的label序号完全相同
        #   即class_names_to_id==mapper.get_coarse2idx()
        # 为了程序的鲁棒性，如果不相同，则应该需要：
        #   1. 先使用LidarsegClassMapper根据每个点的label获取fine class name
        #   2. 再根据fine class name获取每个点的coarse class name
        #   3. 最后根据coarse class name从NuScenesBase定义class_names中获取序号作为label
        # 但这里就不写了233333，进行了一个人工的鲁棒（不是
        if class_names_to_id == mapper.get_coarse2idx():
            print("TRUE: class_names_to_id==mapper.get_coarse2idx()")
        else:
            print("FALSE: class_names_to_id!=mapper.get_coarse2idx()")

        from collections import Counter
        print(Counter(seg_labels))

        # convert to relative path
        lidar_path = lidar_path.replace(root_dir + '/', '')
        cam_path = cam_path.replace(root_dir + '/', '')

        # transpose to yield shape (num_points, 3)
        pts = pts.T

        # append data to train, val or test list in pkl_dict
        data_dict = {
            'points': pts,   ## (N', 3)
            'seg_labels': seg_labels,
            'points_img': pts_img,  # row, col format, shape: (num_points, 2)
            'lidar_path': lidar_path,
            'camera_path': cam_path,
            'boxes': boxes_lidar,
            "sample_token": sample["token"],
            "scene_name": curr_scene_name,
            "calib": calib_infos
        }
        pkl_dict[curr_split].append(data_dict)

    # save to pickle file
    save_dir = osp.join(out_dir, 'lidarseg_preprocess')
    os.makedirs(save_dir, exist_ok=True)
    for split_name in split_names:
        save_path = osp.join(save_dir, '{}{}.pkl'.format(split_name, '_' + subset_name if subset_name else ''))
        with open(save_path, 'wb') as f:
            pickle.dump(pkl_dict[split_name], f)
            print('Wrote preprocessed data to ' + save_path)


if __name__ == '__main__':
    # root_dir = '/datasets_master/nuscenes'
    # out_dir = '/datasets_local/datasets_mjaritz/nuscenes_preprocess'
    root_dir = '/media/lxf/Data/nuScenes'
    out_dir = '/home/lxf/Workspace/Science/xMUDA/xmuda/xmuda/data/nuscenes'

    # nusc = NuScenes(version='v1.0-trainval', dataroot=root_dir, verbose=True)
    # for faster debugging, the script can be run using the mini dataset
    nusc = NuScenes(version='v1.0-mini', dataroot=root_dir, verbose=True)
    # We construct the splits by using the meta data of NuScenes:
    # USA/Singapore: We check if the location is Boston or Singapore.
    # Day/Night: We detect if "night" occurs in the scene description string.
    preprocess(nusc, ['train', 'test'], root_dir, out_dir, location='boston', subset_name='usa')
    preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, location='singapore', subset_name='singapore')
    preprocess(nusc, ['train', 'test'], root_dir, out_dir, keyword='night', keyword_action='exclude', subset_name='day')
    preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, keyword='night', keyword_action='filter', subset_name='night')
