_base_ = [
    '../_base_/datasets/duo_detection.py',
    '../_base_/schedules/schedule_duo.py',
    '../_base_/models/fcos_r50-caffe_fpn.py'
]

# model settings
model = dict(

    neck=dict(
        type="FPNWithAttention",
        attention_cfg=dict(
            type="SDFA",
            in_channels=256
            ),
        attention_stage = "output"
    ),
    bbox_head=dict(
        num_classes=4
    )
)


