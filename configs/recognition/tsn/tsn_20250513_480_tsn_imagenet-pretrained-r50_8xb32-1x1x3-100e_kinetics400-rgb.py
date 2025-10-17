

_base_ = [
    '../../_base_/models/tsn_r50.py', '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'mmaction2/data/CR-5stage/resized_videos'
data_root_val = 'mmaction2/data/CR-5stage-val/resized_videos'
ann_file_train = 'mmaction2/data/CR-5stage/resized_videos/video_labels.txt'
ann_file_val = 'mmaction2/data/CR-5stage-val/resized_videos/val_label.txt'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    # dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=3),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=3),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
#test_evaluator = val_evaluator
test_evaluator = [
    dict(type='AccMetric'),
    dict(type='ConfusionMatrix'),
    dict(type='mAP', average='macro'),  # 多类别AP
    dict(type='AggregateMetric', metrics=[
        dict(type='Accuracy', name='top1'),
        dict(type='Accuracy', topk=3, name='top3')
    ])
]
#default_hooks = dict(checkpoint=dict(interval=3, max_keep_ckpts=3))
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1))
#train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
param_scheduler = [
    # 前5个epoch线性预热
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    # 后续余弦退火
    dict(
        type='CosineAnnealingLR',
        T_max=35,  # 50总epoch-5预热epoch
        by_epoch=True,
        begin=5,       
        end=40)
]
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005, # 将 0.01 修改为 0.005
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (32 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=256)
model = dict(
    cls_head=dict(num_classes=5))
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth'
