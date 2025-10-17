_base_ = ['slowonly_r50_8xb16-4x16x1-256e_kinetics400-rgb.py']

model = dict(
    backbone=dict(
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
    ),
    cls_head=dict(num_classes=700))

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'mmaction2/data/train/soccer_480_train_new'
data_root_val = 'mmaction2/data/val/soccer_480_val_new'
ann_file_train = 'mmaction2/data/train/soccer_480_train_new/train_label.txt'
ann_file_val = 'mmaction2/data/val/soccer_480_val_new/val_label.txt'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=4, frame_interval=16, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=4,
        frame_interval=16,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=4,
        frame_interval=16,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=16,
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


val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning policy
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
    optimizer=dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (16 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=256)
