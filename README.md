# Enhancing 3D Object Detection with Dynamic Dual Teacher Matching
Siqi Xu, Shixuan Xu† , Jing Chen *

！[image](teaser.png)
## Abstract
Existing point-based 3D object detection methods heavily depend on vast
amounts of strongly labeled data, which is both costly and labor-intensive to
acquire. In this paper, we introduce a novel method for 3D object detection called
DDTM (Dynamic Dual Teacher Matching). In the teacher-student framework,
the quality of pseudo-labels significantly influences model performance. However,
during the mutual learning process, detecting errors in pseudo-labels proves challlenging, adversely affecting the quality of subsequent pseudo-label generation.
To address this, we propose a dynamic dual-teacher approach that alternates
between the student and a second teacher, leveraging discrepancies for effective matching and filtering to enhance pseudo-label quality. Furthermore, a fixed
matching threshold can negatively impact the quantity and quality of pseudolabels during continual model optimization. To mitgate this, we implement
dynamic thresholds that adapt to the model’s requirements. We evaluate our
method on the KITTI dataset, consistently outperform state-of-the-art methods,
achieving a 7.7 mAP improvement on a 2% subset of the dataset.
## Environment Setup & Data Setup
[Detmatch](https://github.com/Divadi/DetMatch/blob/main/README.md)
## Pretrain
`./tools/dist_train.sh configs/DDTM/001/pretrain_frcnn/split_0.py 3 --gpus 3 --autoscale-lr`
`./tools/dist_train.sh configs/DDTM/001/pretrain_frcnn/split_1.py 3 --gpus 3 --autoscale-lr`
`./tools/dist_train.sh configs/DDTM/001/pretrain_frcnn/split_2.py 3 --gpus 3 --autoscale-lr`

`./tools/dist_train.sh configs/DDTM/001/pretrain_pvrcnn/split_0.py 3 --gpus 3 --autoscale-lr`
`./tools/dist_train.sh configs/DDTM/001/pretrain_pvrcnn/split_1.py 3 --gpus 3 --autoscale-lr`
`./tools/dist_train.sh configs/DDTM/001/pretrain_pvrcnn/split_2.py 3 --gpus 3 --autoscale-lr`
## Train
`./tools/dist_train.sh configs/DDTM/001/confthr_pvrcnn/split_0.py 3 --gpus 3 --autoscale-lr`
`./tools/dist_train.sh configs/DDTM/001/confthr_pvrcnn/split_1.py 3 --gpus 3 --autoscale-lr`
`./tools/dist_train.sh configs/DDTM/001/confthr_pvrcnn/split_2.py 3 --gpus 3 --autoscale-lr`
## Evaluation
### Average metrics for DDMT (Both modalities)
`python tools/average_runs.py --type fusion --log_jsons outputs/DDTM/001/confthr_pvrcnn/split_0 outputs/DDTM/001/confthr_pvrcnn/split_1 outputs/DDTM/001/confthr_pvrcnn/split_2`

