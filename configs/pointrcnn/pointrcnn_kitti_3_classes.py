outputs_dir = 'outputs/pointrcnn/'
batch_size = 8
work_dir = outputs_dir+'point_rcnn'+'test_debug'
dataset_repeat_multiplier = 5
load_from = None
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
###############################################################################
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
voxel_size = [0.05, 0.05, 0.1]
model = dict(
    type = "OpenPCDetDetector",
    dataset_fields=dict(
        class_names=class_names,
        point_feature_encoder=dict(num_point_features=4),
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        depth_downsample_factor=None),

)