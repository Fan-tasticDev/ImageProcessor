import numpy as np
from PIL import Image

# 用 Pillow 读取并转为 numpy 数组
def load_image(path: str) -> np.ndarray:
    """
    加载图片，返回形状为 (H, W, C) 的 uint8 数组，RGB格式
    """
    img = Image.open(path).convert('RGB')   # 确保统一为RGB
    return np.array(img)

# 根据维度自动保存灰度或彩色图
def save_image(arr: np.ndarray, path: str):
    """
    将 numpy 数组保存为图片。自动处理灰度/彩色
    """
    # 如果是二维灰度图，转为 PIL 'L' 模式
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8), mode='L')
    else:
        img = Image.fromarray(arr.astype(np.uint8))
    img.save(path)
    print(f"图片已保存至：{path}")