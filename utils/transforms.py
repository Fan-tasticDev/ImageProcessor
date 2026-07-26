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



if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from utils.io import load_image
    img = load_image("images/test.jpg")
    gray = rgb_to_gray(img)
    print(f"原图形状: {img.shape}, 灰图形状: {gray.shape}")