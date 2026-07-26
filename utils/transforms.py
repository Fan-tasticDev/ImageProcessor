import numpy as np

def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """
    将 RGB 彩色图像转为灰度图。
    输入: (H, W, 3) uint8
    输出: (H, W) uint8
    """
    # 确保是 RGB 三通道
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("输入必须是三维 RGB 图像")
    # 加权平均法：Y = 0.299R + 0.587G + 0.114B
    gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    return gray.astype(np.uint8)

# 直接 255 - pixel，彩色图每个通道同时反转
def invert(image: np.ndarray) -> np.ndarray:
    """
    图像反转（负片效果）
    支持灰度图 (H,W) 和彩色图 (H,W,3)
    """
    return 255 - image

# 利用 np.where 条件赋值，只有灰度图才能执行
def binarize(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    二值化处理（输入必须是灰度图）
    大于阈值的像素设为255，否则为0
    """
    if image.ndim != 2:
        raise ValueError("二值化仅支持单通道灰度图，请先转换为灰度")
    binary = np.where(image > threshold, 255, 0)
    return binary.astype(np.uint8)

# 像素乘以系数，np.clip 保证范围
def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """
    亮度调整，factor >1 变亮，<1 变暗
    """
    adjusted = image * factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)

# 以 128 为中心扩大或缩小差值，仅演示灰度图处理，确保灰度输入
def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """
    对比度调整（仅对灰度图操作，彩色图需先转灰度或逐通道处理，这里演示灰度）
    formula: (pixel - 128) * factor + 128
    """
    if image.ndim == 3:
        raise ValueError("对比度调整仅支持灰度图，请先转为灰度")
    adjusted = (image.astype(np.float32) - 128) * factor + 128
    return np.clip(adjusted, 0, 255).astype(np.uint8)