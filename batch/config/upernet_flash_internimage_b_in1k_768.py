_base_ = [
    './_base_/schedule_base.py',
    './_base_/default_runtime.py',
]
custom_imports = dict(
    imports=['mmseg.models.losses.contrastive_loss'],
    allow_failed_imports=False
)
num_classes = 5
crop_size = (768, 768)
data_preprocessor = dict(
    type='SegDataPreProcessor', _scope_='mmseg',
    #type='CustomSegDataPreProcessor', _scope_='mmseg',
    size=crop_size,
    # mean=[120.89796271 ,95.93733616 ,102.15402776],
    # std=[42.58632222, 42.28617965 ,42.23552982],  
    mean=[137.8555444 , 114.66734371 ,127.38273215],
    std=[18.02458473 ,17.13238506, 14.84981372],#mos2
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=64),
)


model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ext-FlashInternImage',
        channels=112,
      core_op='DCNv4',
        depths=[4, 4, 21, 4, ],
        drop_path_rate=0.3,
        dw_kernel_size=3,
        groups=[7, 14, 28, 56, ],
        init_cfg=dict(
            checkpoint="/home/yansu/mmlabmat/segmantation/mmseg-extension-test/pretrained/flash_internimage/flash_intern_image_b_1k_224.pth",
            # 'https://huggingface.co/OpenGVLab/DCNv4/resolve/main/flash_intern_image_b_1k_224.pth',
            type='Pretrained'),
        layer_scale=1.0,
        mlp_ratio=4.0,
        norm_layer='LN',
        offset_scale=0.5,
        out_indices=(0, 1, 2, 3,),
        post_norm=True,
        with_cp=False),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[112, 224, 448, 896, ],
        in_index=[0, 1, 2, 3, ],
        enable_feature_enhancement=False,
        enhancement_class_pairs=[[1, 2]], # 需要增强区分性的类别对
        enhancement_alpha=0.2,  # 增强强度
        log_enhancement=True,   # 是否记录增强效果
        loss_decode=dict(
            #losses=[
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.02,
                #dict(type='CenterLoss', num_classes=num_classes, feat_dim=512, loss_weight=0.1)
            #]
        ),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
       num_classes=num_classes,
        pool_scales=(1, 2, 3, 6,),
        type='UPerHead'),
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=448,
        in_index=2,
        loss_decode=[dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False,
            ),

        ],
        # 特征增强相关参数
        enable_feature_enhancement=False,
        enhancement_class_pairs=[[1, 2]], # 需要增强区分性的类别对
        enhancement_alpha=0.2,  # 增强强度
        log_enhancement=True,   # 是否记录增强效果
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=num_classes,
        num_convs=1,
        type='FCNHead'),
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'),
    # test_cfg=dict(crop_size=(3072, 3072,), mode='slide', stride=(2560, 2560,)),
    test_cfg=dict(crop_size=(2048, 2048,), mode='slide', stride=(2048-128, 2048-128,)),
)

# By default, models are trained on 2 GPUs with 4 images per GPU
# train_dataloader = dict(num_batch_per_epoch=5)
# val_dataloader = dict(num_batch_per_epoch=5)
# test_dataloader = val_dataloader

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend', save_dir='./runsbatch/graphene_batch1_FlashInternImage_b/local'),
        # dict(type='TensorboardVisBackend', save_dir='runs/tb'),
        dict(type='MLflowVisBackend', save_dir='./runsbatch/graphene_batch1_FlashInternImage_b/mlruns',
             exp_name='batch1',
             run_name='FlashInternImage_b',
             ),
    ])

