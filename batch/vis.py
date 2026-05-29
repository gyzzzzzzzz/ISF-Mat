# import argparse
# import os
# import os.path as osp
# import torch
# import mmcv
# import cv2
# import numpy as np
# from mmengine.config import Config, DictAction
# from mmengine.logging import print_log
# from mmseg.apis import init_model, inference_model

# # Import your custom modules
# import mmsegext
# import mmseg2dmat

# def parse_args():
#     parser = argparse.ArgumentParser(description='Test a segmentor and visualize results')
#     parser.add_argument('--config', help='test config file path')
#     parser.add_argument('--config-merge', help='merge config file path')
#     parser.add_argument('--checkpoint', help='checkpoint file path')
#     parser.add_argument('--input-dir', help='input image directory')
#     parser.add_argument('--output-dir', default='visualization_results', help='output directory')
#     parser.add_argument('--device', default='cuda:0', help='Device used for inference')
#     parser.add_argument(
#         '--cfg-options',
#         nargs='+',
#         action=DictAction,
#         help='override some settings in the used config')
#     args = parser.parse_args()
#     return args

# def add_test_pipeline_if_missing(cfg):
#     """Add test_pipeline to config if it doesn't exist"""
#     if not hasattr(cfg, 'test_pipeline') or cfg.test_pipeline is None:
#         # 创建一个基本的test_pipeline
#         test_pipeline = [
#             dict(type='LoadImageFromFile'),
#             dict(type='Resize', scale=(512, 512), keep_ratio=True),
#             dict(type='LoadAnnotations', reduce_zero_label=False),
#             dict(type='PackSegInputs')
#         ]
#         cfg.test_pipeline = test_pipeline
#         print_log('Added default test_pipeline to config', logger='current')
#     return cfg

# def main():
#     args = parse_args()
    
#     # Load configuration
#     cfg = Config.fromfile(args.config)
    
#     # Merge additional config if provided
#     if args.config_merge is not None:
#         cfg_merge = Config.fromfile(args.config_merge)
#         cfg.merge_from_dict(cfg_merge.to_dict())
    
#     # Apply any additional configuration options
#     if args.cfg_options is not None:
#         cfg.merge_from_dict(args.cfg_options)
    
#     # Add test_pipeline if missing
#     cfg = add_test_pipeline_if_missing(cfg)
    
#     # Build the model and load checkpoint
#     model = init_model(cfg, args.checkpoint, device=args.device)
#     print_log(f'Model loaded successfully from {args.checkpoint}', logger='current')
    
#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     # Define color mapping for visualization
#     colors = {
#         0: [255, 255, 255],        # 背景-白色
#         1: [0,0,205][::-1],       # 1类-深蓝色 (RGB转BGR)
#         2: [135,206,235][::-1],   # 2类-浅蓝色 (RGB转BGR)
#     }
    
#     # Get image files from input directory or dataset config
#     if args.input_dir and os.path.exists(args.input_dir):
#         # Use specified input directory
#         img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
#         img_files = []
#         for ext in img_extensions:
#             img_files.extend([f for f in os.listdir(args.input_dir) if f.lower().endswith(ext)])
#         img_paths = [osp.join(args.input_dir, f) for f in img_files]
#     else:
#         # Try to get from dataset config
#         try:
#             from mmseg.registry import DATASETS
#             test_dataset = DATASETS.build(cfg.test_dataloader.dataset)
#             img_paths = []
#             for i in range(len(test_dataset)):
#                 data = test_dataset[i]
#                 img_paths.append(data['img_path'])
#         except Exception as e:
#             print_log(f'Failed to load dataset: {e}', logger='current')
#             print_log('Please specify --input-dir', logger='current')
#             return
    
#     # Process each image
#     print_log(f'Starting visualization on {len(img_paths)} images', logger='current')
    
#     for i, img_path in enumerate(img_paths):
#         if i % 10 == 0:
#             print_log(f'Processing image {i}/{len(img_paths)}', logger='current')
        
#         img_name = osp.splitext(osp.basename(img_path))[0]
        
#         # Read original image to get dimensions
#         ori_img = mmcv.imread(img_path)
#         h, w = ori_img.shape[:2]
        
#         # Run inference
#         try:
#             result = inference_model(model, img_path)
#         except Exception as e:
#             print_log(f'Inference failed for {img_path}: {e}', logger='current')
#             continue
        
#         # Extract prediction mask correctly based on the result type
#         if isinstance(result, list):
#             # If result is a list, take the first item
#             pred_item = result[0]
#         else:
#             pred_item = result
            
#         # Check for different result formats and extract the segmentation mask
#         if hasattr(pred_item, 'pred_sem_seg') and pred_item.pred_sem_seg is not None:
#             pred_mask = pred_item.pred_sem_seg.data.cpu().numpy().squeeze()
#         elif hasattr(pred_item, 'seg_logits') and pred_item.seg_logits is not None:
#             pred_mask = pred_item.seg_logits.data.cpu().numpy().argmax(axis=0)
#         elif isinstance(pred_item, dict) and 'pred_sem_seg' in pred_item:
#             pred_mask = pred_item['pred_sem_seg'].data.cpu().numpy().squeeze()
#         elif isinstance(pred_item, dict) and 'seg_pred' in pred_item:
#             pred_mask = pred_item['seg_pred'].data.cpu().numpy().squeeze()
#         elif isinstance(pred_item, torch.Tensor):
#             pred_mask = pred_item.cpu().numpy().squeeze()
#             if len(pred_mask.shape) > 2:
#                 pred_mask = pred_mask.argmax(axis=0)
#         else:
#             print_log(f'Unsupported result format for {img_path}. Skipping.', logger='current')
#             continue
        
#         # Resize mask to original size if needed
#         if pred_mask.shape[:2] != (h, w):
#             pred_mask = cv2.resize(pred_mask.astype(np.uint8), (w, h), 
#                                 interpolation=cv2.INTER_NEAREST)
        
#         # Create colored mask
#         colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
#         for layer_num, color in colors.items():
#             colored_mask[pred_mask == layer_num] = color
        
#         # Save colored mask directly to output directory
#         mask_path = osp.join(args.output_dir, f'{img_name}_mask.png')
#         cv2.imwrite(mask_path, colored_mask)
        
#         # Free memory
#         torch.cuda.empty_cache()
    
#     print_log(f'Visualization completed. Results saved to {args.output_dir}', logger='current')

# if __name__ == '__main__':
#     main()




import json
import numpy as np
import cv2
from pathlib import Path
from pycocotools import mask as maskUtils

# ========== 配置 ==========
COCO_JSON = "/home/guyizhan/datasets/fscil_data/Graphene/annotations/instances_val2024.json"   # 你的 COCO 文件
OUTPUT_DIR = "/home/guyizhan/contrast_experiment/GrapheneGTVAL"   # 输出彩色掩码目录
CATEGORY_TO_ID = {
    "thinlayer": 1,   # category_name -> class_id (用于 mask 像素值)
    "monolayer": 2,
}
IGNORE_CATEGORIES = []  # 要忽略的类别名（如 "thicklayer", "impurity"）
# 颜色 (BGR, OpenCV)
COLOR_MAP = {
    0: (255, 255, 255),   # 背景 -> 白色
    1: (205, 0, 0),       # 1类 (monolayer) -> 深蓝色 (RGB: 0,0,205 转 BGR)
    2: (235, 206, 135),   # 2类 (thinlayer) -> 浅蓝色 (RGB: 135,206,235 转 BGR)
}
# =========================

def coco_to_colored_masks(coco_json, output_dir, cat_to_id, color_map, ignore_cats=[]):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(coco_json, 'r') as f:
        coco = json.load(f)
    
    # 建立 category_name -> category_id 映射
    cat_name_to_id = {cat['name']: cat['id'] for cat in coco['categories']}
    target_cat_ids = {cat_name_to_id[name]: cid for name, cid in cat_to_id.items() if name not in ignore_cats}
    
    # 按 image_id 分组 annotations
    img_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        img_anns.setdefault(img_id, []).append(ann)
    
    for img_info in coco['images']:
        img_id = img_info['id']
        h, w = img_info['height'], img_info['width']
        mask = np.zeros((h, w), dtype=np.uint8)  # 背景 0
        
        if img_id in img_anns:
            for ann in img_anns[img_id]:
                cat_id = ann['category_id']
                if cat_id not in target_cat_ids:
                    continue
                class_id = target_cat_ids[cat_id]   # 1 或 2
                seg = ann['segmentation']
                if isinstance(seg, list):
                    # 多边形
                    for poly in seg:
                        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(mask, [pts], class_id)
                elif isinstance(seg, dict) and 'counts' in seg:
                    # RLE
                    rle = maskUtils.frPyObjects(seg, h, w)
                    ann_mask = maskUtils.decode(rle)
                    mask[ann_mask > 0] = class_id
                else:
                    print(f"Unsupported seg type: {type(seg)}")
        
        # 转换为彩色 BGR 图像
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        for val, bgr in color_map.items():
            color_img[mask == val] = bgr
        
        out_path = output_dir / f"{Path(img_info['file_name']).stem}_color.png"
        cv2.imwrite(str(out_path), color_img)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    coco_to_colored_masks(COCO_JSON, OUTPUT_DIR, CATEGORY_TO_ID, COLOR_MAP, IGNORE_CATEGORIES)