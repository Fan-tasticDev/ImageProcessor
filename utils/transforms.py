import numpy as np
from scipy.ndimage import convolve, sobel


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
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
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


# 高斯核生成函数
def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    生成高斯核（二维）
    size: 核大小（奇数）
    sigma: 标准差
    """
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


# 高斯核生成函数
def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    生成高斯核（二维）
    size: 核大小（奇数）
    sigma: 标准差
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

# 高斯模糊
def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    对灰度或彩色图像进行高斯模糊
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    if image.ndim == 3:
        # 彩色图：分通道处理
        blurred = np.zeros_like(image)
        for c in range(image.shape[2]):
            blurred[:,:,c] = convolve(image[:,:,c], kernel)
        return blurred.astype(np.uint8)
    else:
        return convolve(image, kernel).astype(np.uint8)

# 锐化滤镜
def sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    拉普拉斯锐化
    公式：sharpened = original + strength * (original - blurred)
    """
    blurred = gaussian_blur(image, kernel_size=5, sigma=1.0)
    # 减去模糊的部分得到边缘，再加回原图
    if image.ndim == 3:
        detail = image.astype(np.float32) - blurred.astype(np.float32)
        sharpened = image.astype(np.float32) + strength * detail
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    else:
        detail = image.astype(np.float32) - blurred.astype(np.float32)
        sharpened = image.astype(np.float32) + strength * detail
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
# 简化版 Canny 边缘检测
def canny_edge(image: np.ndarray, low_threshold: float = 50, high_threshold: float = 100) -> np.ndarray:
    """
    简化版Canny边缘检测（仅支持灰度图）
    - 使用Sobel算子计算梯度和方向
    - 非极大值抑制（简化版，仅保留局部最大值）
    - 双阈值处理
    """
    if image.ndim == 3:
        raise ValueError("Canny仅支持单通道灰度图")
    
    # 1. 高斯平滑
    blurred = gaussian_blur(image, kernel_size=5, sigma=1.0)
    
    # 2. 计算梯度（使用Sobel算子）
    grad_x = sobel(blurred, axis=1)  # 水平梯度
    grad_y = sobel(blurred, axis=0)  # 垂直梯度
    magnitude = np.hypot(grad_x, grad_y)     # 梯度幅值
    magnitude = magnitude / np.max(magnitude) * 255
    direction = np.arctan2(grad_y, grad_x)   # 方向角（弧度）
    
    # 3. 非极大值抑制（简化版：在3x3邻域内仅保留最大值）
    suppressed = np.zeros_like(magnitude)
    rows, cols = magnitude.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            # 量化方向为4个主方向（0, 45, 90, 135度）
            angle = direction[i, j] * 180 / np.pi
            angle = angle % 180
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [magnitude[i, j+1], magnitude[i, j-1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [magnitude[i+1, j], magnitude[i-1, j]]
            else: # 112.5 <= angle < 157.5
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
            
            if magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = magnitude[i, j]
    
    # 4. 双阈值处理
    strong = 255
    weak = 75
    edge_map = np.zeros_like(suppressed, dtype=np.uint8)
    strong_i, strong_j = np.where(suppressed >= high_threshold)
    weak_i, weak_j = np.where((suppressed >= low_threshold) & (suppressed < high_threshold))
    
    edge_map[strong_i, strong_j] = strong
    edge_map[weak_i, weak_j] = weak
    
    # 弱边缘连接（简化：若弱边缘像素周围8邻域有强边缘，则保留）
    for i, j in zip(weak_i, weak_j):
        if ((i > 0 and i < rows-1) and (j > 0 and j < cols-1)):
            window = edge_map[i-1:i+2, j-1:j+2]
            if np.any(window == strong):
                edge_map[i, j] = strong
            else:
                edge_map[i, j] = 0
    return edge_map