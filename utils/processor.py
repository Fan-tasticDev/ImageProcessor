import numpy as np
from . import transforms

class ImageProcessor:
    def __init__(self, image: np.ndarray):
        """
        image: 任意 numpy 数组图像 (H,W) 或 (H,W,3)
        """
        self.original = image.copy()
        self.current = image.copy()
        self._gray_flag = False  # 用于记录当前是否已转为灰度

    def _ensure_gray(self):
        """内部方法：确保 current 是灰度图，如果不是则自动转换"""
        if self.current.ndim == 3:
            self.current = transforms.rgb_to_gray(self.current)
            self._gray_flag = True

    def apply(self, func, *args, **kwargs):
        """应用一个变换函数，返回 self 以支持链式调用"""
        self.current = func(self.current, *args, **kwargs)
        return self

    def reset(self):
        """恢复到原始图像"""
        self.current = self.original.copy()
        self._gray_flag = False
        return self

    def to_gray(self):
        """转为灰度"""
        self.apply(transforms.rgb_to_gray)
        self._gray_flag = True
        return self

    def invert(self):
        """反转颜色"""
        self.apply(transforms.invert)
        return self

    def binarize(self, threshold=128):
        """二值化，自动先转灰度"""
        self._ensure_gray()
        self.apply(transforms.binarize, threshold)
        return self

    def brightness(self, factor):
        """调整亮度"""
        self.apply(transforms.adjust_brightness, factor)
        return self

    def contrast(self, factor):
        """调整对比度（自动转灰度）"""
        self._ensure_gray()
        self.apply(transforms.adjust_contrast, factor)
        return self

    def get_image(self):
        return self.current