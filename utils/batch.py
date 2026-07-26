import os
import numpy as np
from .io import load_image, save_image
from .processor import ImageProcessor

# 操作名映射表
OPERATIONS_MAP = {
    "gray": "to_gray",
    "invert": "invert",
    "binarize": "binarize",
    "brightness": "brightness",
    "contrast": "contrast",
    "blur": "blur",
    "sharpen": "sharpen",
    "canny": "canny",
    "equalize": "equalize_hist",
}

def batch_process(input_dir: str,
                  output_dir: str,
                  operations: list,
                  params: dict = None,
                  extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')):
    """
    批量处理文件夹内的图片。
    
    参数:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径（不存在则自动创建）
        operations: 操作名称列表，按顺序执行，如 ['gray', 'equalize', 'sharpen']
        params: 字典，键为操作名，值为参数字典，如 {'binarize': {'threshold': 100}}
        extensions: 支持处理的图片扩展名元组
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    if params is None:
        params = {}
    
    # 遍历输入文件夹
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(extensions):
            continue
        
        input_path = os.path.join(input_dir, filename)
        name, ext = os.path.splitext(filename)
        
        print(f"正在处理: {filename} ...", end=" ")
        
        try:
            # 加载图片
            img = load_image(input_path)
            proc = ImageProcessor(img)
            
            # 依次应用操作
            for op_name in operations:
                if op_name not in OPERATIONS_MAP:
                    print(f"警告: 未知操作 '{op_name}'，跳过")
                    continue
                
                method_name = OPERATIONS_MAP[op_name]
                method = getattr(proc, method_name)
                
                # 获取该操作的参数
                kwargs = params.get(op_name, {})
                method(**kwargs)
            
            # 保存结果
            result = proc.get_image()
            output_path = os.path.join(output_dir, f"{name}_processed{ext}")
            save_image(result, output_path)
            print("完成")
            
        except Exception as e:
            print(f"失败: {e}")