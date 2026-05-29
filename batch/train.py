
import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import os.path as osp
import sys
from pprint import pprint
import copy
from mmseg.registry import MODELS
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
import mmsegext
import mmseg2dmat
import random
import mmcv
from mmcv.transforms import BaseTransform, TRANSFORMS
from mmseg.models import SegDataPreProcessor

# Add this import at the top
from mmengine.hooks import Hook
from mmseg.registry import HOOKS



# 添加统计参数量的函数
def count_parameters(model):
    """统计模型参数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params





# 自定义 DebugCheckImage 类
@TRANSFORMS.register_module()
class DebugCheckImage:
    def __call__(self, results):
        img = results['img']
        print(f"[Debug] Image range: [{img.min()}, {img.max()}]")
        return results
@MODELS.register_module()
class CustomSegDataPreProcessor(SegDataPreProcessor):
    def forward(self, data, training=False):
        # 提取 data_samples
        data_samples = data['data_samples']
        if training:
        # 检查并提取 teacher_outputs
            if 'teacher_outputs' in data:
                teacher_outputs = data.pop('teacher_outputs')  # 从 data 中移除 teacher_outputs
            # 确保 teacher_outputs 的长度与 data_samples 一致
                assert len(teacher_outputs) == len(data_samples), "Mismatch in lengths"
            # 将 teacher_outputs 添加到每个 data_sample 的 metainfo 中
                for i, data_sample in enumerate(data_samples):
                    data_sample.set_metainfo({'teacher_outputs': teacher_outputs[i]})
            

        
        # 调用父类的 forward 方法
        data = super().forward(data, training)

        return data

@TRANSFORMS.register_module()
class BASETransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def transform(self, results: dict) -> dict:
        
        # 检查语义分割掩码是否在 results 中
        if 'gt_seg_map' in results:
            # 提取语义分割掩码
            gt_seg_map = results['gt_seg_map']
            
            gt_seg_map[gt_seg_map == 2] = 255
            # 更新results中的gt_seg_map
            results['gt_seg_map'] = gt_seg_map
        else:
            print("No gt_seg_map found in results.")  # 当找不到掩码时，打印信息
        
        return results


@TRANSFORMS.register_module()
class NEWTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def transform(self, results: dict) -> dict:
        
        # 检查语义分割掩码是否在 results 中
        if 'gt_seg_map' in results:
            # 提取语义分割掩码
            gt_seg_map = results['gt_seg_map']
            
            #gt_seg_map[gt_seg_map == 1] = 255
            # 更新results中的gt_seg_map
            results['gt_seg_map'] = gt_seg_map
        else:
            print("No gt_seg_map found in results.")  # 当找不到掩码时，打印信息
        
        return results


debug_cfg = dict(
    visualizer=dict(vis_backends=[
        dict(type='LocalVisBackend', save_dir='runs_temp/local/temp'),
        dict(type='MLflowVisBackend', save_dir='runs_temp/mlruns', exp_name='exp_name', run_name='run_name', ), ]),
    work_dir='runs_temp/work_dir',
    train_dataloader=dict(num_batch_per_epoch=11),
    val_dataloader=dict(num_batch_per_epoch=11),
    train_cfg=dict(type='IterBasedTrainLoop', max_iters=40, val_interval=1, val_begin=0)
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--debug', type=bool, default=False, help='debug')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--config-merge', help='merge config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    print('=' * 20, ' Argument ', '>' * 20)
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    print('<' * 20, ' Argument ', '=' * 20)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    if args.config_merge is not None:
        cfg_merge = Config.fromfile(args.config_merge)
        cfg.merge_from_dict(cfg_merge.to_dict())

    if args.cfg_options is not None:
        cfg_options_copy = copy.deepcopy(args.cfg_options)
        for k, v in args.cfg_options.items():
            if 'cfg.' in k:
                cfg_options_copy.pop(k)
                print(type(k), type(v))
                if isinstance(v, str):
                    exec(f"{k}='{v}'")
                    print(f"{k}='{v}'")
                else:
                    exec(f"{k}={v}")
                    print(f"{k}={v}")
        print(cfg_options_copy)
        cfg.merge_from_dict(cfg_options_copy)

    # 添加验证前保存权重的Hook
    if not hasattr(cfg, 'custom_hooks'):
        cfg.custom_hooks = []
    
    # # 添加验证前保存权重的Hook
    # save_hook = dict(
    #     type='SaveBeforeValidationHook',
    #     save_dir=None,  # 使用runner的work_dir
    #     interval=cfg.train_cfg.val_interval,  # 与验证间隔一致
    #     save_optimizer=True,  # 同时保存优化器状态
    #     filename_tmpl='before_val_iter_{}.pth',  # 文件名模板
    #     max_keep=1  # 只保留最近的3个保存点，避免占用过多磁盘空间
    # )
    # cfg.custom_hooks.append(save_hook)




    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'
    if args.debug:
        cfg.merge_from_dict(debug_cfg)

    # resume training
    cfg.resume = args.resume

    # pprint(cfg.to_dict())
    if args.cfg_options is not None:
        for k, v in args.cfg_options.items():
            if 'cfg.' in k:
                c = f'print(\'{k}\',\':\',type({k}),\'=\',{k})'
                exec(c)
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    # model=runner.model
    # model.load_state_dict()
    # start training

    # 在训练前统计并打印模型参数量
    model = runner.model
    total_params, trainable_params = count_parameters(model)
    
    # 打印参数统计信息
    print_log(f'模型总参数量: {total_params:,}', logger='current')
    print_log(f'模型可训练参数量: {trainable_params:,}', logger='current')
    print_log(f'不可训练参数量: {total_params - trainable_params:,}', logger='current')
    print_log(f'可训练参数比例: {trainable_params/total_params*100:.2f}%', logger='current')
   




    model=runner.train()


if __name__ == '__main__':

    pprint(sys.argv)
    if sys.argv.__len__() == 1:
        params = {
        
        }
        cfg_options = [
# "--config", "/home/guyizhan/FlashIterImage/batch/config/upernet_flash_internimage_b_in1k_768.py",
#                 "--config-merge", "/home/guyizhan/FlashIterImage/batch/config/dataset/2024_annlab_graphene_batch5_768.py",
#                 "--work-dir", "/home/guyizhan/work_dirs/three/work_dirs",
#                 "--cfg-options", "cfg.visualizer.vis_backends[0].save_dir=/home/guyizhan/work_dirs/three/local",
#                 "--cfg-options", "cfg.visualizer.vis_backends[1].save_dir=/home/guyizhan/work_dirs/three/mlruns",
#                 "--cfg-options", "cfg.visualizer.vis_backends[1].exp_name=batch5",
#                 "--cfg-options", "cfg.visualizer.vis_backends[1].run_name=5/FlashInternImage",
#                 "--cfg-options", "cfg.param_scheduler[1].end=130000",
#                 "--cfg-options", "cfg.train_cfg.max_iters=130000",
#                 "--cfg-options", "cfg.train_cfg.val_interval=100",
#                 "--cfg-options", "cfg.default_hooks.checkpoint.interval=100"
#         ]
        

# "--config", "/home/guyizhan/FSS-master/pretrained/upernet_flash_internimage_b_in1k_768.py",
#                 "--config-merge", "/home/guyizhan/FlashIterImage/batch/config/dataset/2024_annlab_graphene_batch5_768.py",
#                 "--work-dir", "result/thin/graphene/graphene_batch5_FlashInternImage/work_dirs",
#                 "--cfg-options", "cfg.visualizer.vis_backends[0].save_dir=result/thin/graphene/graphene_batch5_FlashInternImage/local",
#                 "--cfg-options", "cfg.visualizer.vis_backends[1].save_dir=result/thin/graphene/graphene_batch5_FlashInternImage/mlruns",
#                 "--cfg-options", "cfg.visualizer.vis_backends[1].exp_name=batch5",
#                 "--cfg-options", "cfg.visualizer.vis_backends[1].run_name=5/FlashInternImage",
#                 "--cfg-options", "cfg.param_scheduler[1].end=130000",
#                 "--cfg-options", "cfg.train_cfg.max_iters=130000",
#                 "--cfg-options", "cfg.train_cfg.val_interval=13000",
#                 "--cfg-options", "cfg.default_hooks.checkpoint.interval=13000"
#         ]



#         ]


# "--config", "/home/guyizhan/FlashIterImage/batch/config/upernet_flash_internimage_b_in1k_768.py",
#              "--config-merge", "/home/guyizhan/FSS-master/mmseg_dst/2024_annlab_1-0-Graphene_batch5_768_new.py",
#                 "--work-dir", "try/result/mono/graphene/graphene_batch5_FlashInternImage/work_dirs",
#                 "--cfg-options", "cfg.visualizer.vis_backends[0].save_dir=try/result/mono/graphene/graphene_batch5_FlashInternImage/local",
#                "--cfg-options", "cfg.visualizer.vis_backends[1].save_dir=try/result/mono/graphene/graphene_batch5_FlashInternImage/mlruns",
#                 "--cfg-options", "cfg.visualizer.vis_backends[1].exp_name=batch5",
#                 "--cfg-options", "cfg.visualizer.vis_backends[1].run_name=5/FlashInternImage",
#                 "--cfg-options", "cfg.param_scheduler[1].end=50000",
#                 "--cfg-options", "cfg.train_cfg.max_iters=50000",
#                 "--cfg-options", "cfg.train_cfg.val_interval=5000",
#                 "--cfg-options", "cfg.default_hooks.checkpoint.interval=5000",
#                "--cfg-options","cfg.load_from=/home/guyizhan/result/thin/graphene/graphene_batch5_FlashInternImage/work_dirs/best_mIoU_iter_52000.pth"
#          ]
 #      cfg_options = [
"--config", "/home/guyizhan/FlashIterImage/batch/config/upernet_flash_internimage_b_in1k_768.py",
             "--config-merge", "/home/guyizhan/FSS-master/mmseg_dst/2024_annlab_1-0-MoS2_batch5_768_new.py",
                "--work-dir", "try/result/mono/MoS2/MoS2_batch5_FlashInternImage/work_dirs",
                "--cfg-options", "cfg.visualizer.vis_backends[0].save_dir=try/result/mono/MoS2/MoS2_batch5_FlashInternImage/local",
                "--cfg-options", "cfg.visualizer.vis_backends[1].save_dir=try/result/mono/MoS2/MoS2_batch5_FlashInternImage/mlruns",
                "--cfg-options", "cfg.visualizer.vis_backends[1].exp_name=batch5",
                "--cfg-options", "cfg.visualizer.vis_backends[1].run_name=5/FlashInternImage",
              "--cfg-options", "cfg.param_scheduler[1].end=400",
                "--cfg-options", "cfg.train_cfg.max_iters=400",
                "--cfg-options", "cfg.train_cfg.val_interval=100",
                "--cfg-options", "cfg.default_hooks.checkpoint.interval=100",
                "--cfg-options","cfg.load_from=/home/guyizhan/result/thin/MoS2/MoS2_batch5_FlashInternImage/work_dirs/best_mIoU_iter_91000.pth"
        ]
# "--config", "/home/guyizhan/FlashIterImage/batch/config/upernet_flash_internimage_b_in1k_768.py",
#              "--config-merge", "/home/guyizhan/FSS-master/mmseg_dst/2024_annlab_1-0-MoS2_batch5_768_new.py",
#                 "--work-dir", "try/result/mono/MoS2/MoS2_batch5_FlashInternImage/work_dirs",
#                 "--cfg-options", "cfg.visualizer.vis_backends[0].save_dir=try/result/mono/MoS2/MoS2_batch5_FlashInternImage/local",
#                 "--cfg-options", "cfg.visualizer.vis_backends[1].save_dir=try/result/mono/MoS2/MoS2_batch5_FlashInternImage/mlruns",
#                 "--cfg-options", "cfg.visualizer.vis_backends[1].exp_name=batch5",
#                 "--cfg-options", "cfg.visualizer.vis_backends[1].run_name=5/FlashInternImage",
#               "--cfg-options", "cfg.param_scheduler[1].end=60000",
#                 "--cfg-options", "cfg.train_cfg.max_iters=60000",
#                 "--cfg-options", "cfg.train_cfg.val_interval=5000",
#                 "--cfg-options", "cfg.default_hooks.checkpoint.interval=5000",
#                 "--cfg-options","cfg.load_from=/home/guyizhan/result/thin/MoS2/MoS2_batch5_FlashInternImage/work_dirs/best_mIoU_iter_91000.pth"
# "--config", "/home/guyizhan/FlashIterImage/batch/config/upernet_flash_internimage_b_in1k.py",
#              "--config-merge", "/home/guyizhan/FlashIterImage/batch/config/dataset_medicine/dataset.py",
#                 "--work-dir", "medicine/result/work_dirs",
#                 "--cfg-options", "cfg.visualizer.vis_backends[0].save_dir=medicine/result/local",
#                 "--cfg-options", "cfg.visualizer.vis_backends[1].save_dir=medicine/result/mlruns",
#                 "--cfg-options", "cfg.visualizer.vis_backends[1].exp_name=batch5",
#                 "--cfg-options", "cfg.visualizer.vis_backends[1].run_name=5/FlashInternImage",
#               "--cfg-options", "cfg.param_scheduler[1].end=60000",
#                 "--cfg-options", "cfg.train_cfg.max_iters=60000",
#                 "--cfg-options", "cfg.train_cfg.val_interval=5000",
#                 "--cfg-options", "cfg.default_hooks.checkpoint.interval=5000",
                #"--cfg-options","cfg.load_from=/home/guyizhan/result/thin/MoS2/MoS2_batch5_FlashInternImage/work_dirs/best_mIoU_iter_91000.pth"
#        ]
        
                


        pprint(params)
        for k, v in params.items():
            sys.argv.append(k)
            sys.argv.append(v)
        sys.argv=sys.argv+cfg_options
    main()
