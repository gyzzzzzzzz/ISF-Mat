# 文件名: /home/guyizhan/FlashIterImage/import_all_custom_modules.py

import os
import importlib.util
import glob

def import_module_from_path(file_path):
    try:
        module_name = os.path.basename(file_path).split('.')[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            print(f"无法为 {file_path} 创建 spec")
            return None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"成功导入: {file_path}")
        return module
    except Exception as e:
        print(f"导入 {file_path} 失败: {str(e)}")
        return None

# 自定义组件的根目录
custom_root = "/home/guyizhan/FlashIterImage/zlib/mmseg-extension"

# 导入所有可能包含注册组件的Python文件
component_dirs = [
    # 模型和骨干网络
    f"{custom_root}/mmsegext/models/backbones/*.py",
    f"{custom_root}/mmsegext/models/decode_heads/*.py",
    f"{custom_root}/mmsegext/models/losses/*.py",
    f"{custom_root}/mmsegext/models/necks/*.py",
    f"{custom_root}/mmsegext/models/segmentors/*.py",
    
    # 数据集
    f"{custom_root}/mmsegext/datasets/*.py",
    
    # 评估指标
    f"{custom_root}/mmsegext/evaluation/*.py",
    
    # 任何其他可能包含注册组件的目录
    f"{custom_root}/mmsegext/utils/*.py",
]

print("开始导入自定义模块...")
imported_modules = []

for pattern in component_dirs:
    for file_path in glob.glob(pattern):
        # 跳过 __init__.py 和 __pycache__ 目录
        if "__init__.py" in file_path or "__pycache__" in file_path:
            continue
        
        module = import_module_from_path(file_path)
        if module:
            imported_modules.append(module)

print(f"成功导入 {len(imported_modules)} 个模块")
