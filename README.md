# Dynamic Dual Teacher Matching and Adaptive Thresholds for Enhanced 3D Object Detection
Siqi Xu, Shixuan Xu† , Jing Chen *

![image](moedel.png)
## Abstract
Existing point-based 3D object detection methods heavily rely on vast amounts
of strongly labeled data, which is costly and labor-intensive to acquire. In this
paper, we propose a novel method called Dynamic Dual Teacher Matching
(DDTM) for 3D object detection. DDTM leverages semi-supervised learning to
reduce the dependency on large annotated datasets and improve detection performance by enhancing the quality of pseudo-labels. Our approach introduces a
dynamic dual-teacher strategy that alternates between the student and a second teacher, utilizing discrepancies for matching and filtering to generate more
reliable pseudo-labels. Additionally, to mitigate the impact of a fixed matching
threshold, we introduce adaptive thresholds that adjust based on the model’s
performance. Experimental results on the KITTI dataset demonstrate that our
method consistently outperforms existing methods, achieving a significant 7.7
mAP improvement on a 2% subset of the dataset. Our contributions include
a dual-teacher dynamic matching strategy to improve pseudo-label quality, a
novel dynamic threshold algorithm, and experimental validation of our approach’s
effectiveness and robustness in 3D semi-supervised object detection.

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

## Acknowledgements

We sincerely thank the following excellent works and open-source codebases:

-[DetMatch](https://github.com/Divadi/DetMatch)

-[MMDetection3D](https://github.com/open-mmlab/mmdetection3d)

-[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

-[Spconv](https://github.com/traveller59/spconv)

