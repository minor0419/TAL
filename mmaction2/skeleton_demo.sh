python demo/demo_skeleton.py demo/STGCN.avi demo/skeleton_demo.mp4 --config configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py \
--checkpoint work_dirs/stgcn_80e_ntu60_xsub_keypoint/best_top1_acc_epoch_60.pth \
--det-config demo/faster_rcnn_r50_fpn_2x_coco.py \
--det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
--det-score-thr 0.9 \
--pose-config demo/hrnet_w32_coco_256x192.py \
--pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
--label-map tools/data/skeleton/label_map_ntu120.txt