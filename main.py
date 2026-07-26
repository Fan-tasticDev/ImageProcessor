import matplotlib.pyplot as plt
from utils.io import load_image, save_image
from utils.processor import ImageProcessor

def main():
    # 加载图片
    img_path = "images/test.jpg"
    original = load_image(img_path)

    # 创建处理器实例
    proc = ImageProcessor(original)

    # ---- 测试1：反转 -----
    proc.invert()
    inverted = proc.get_image()
    save_image(inverted, "images/test_invert.jpg")
    proc.reset()   # 恢复原图

    # ---- 测试2：亮度调整（变亮1.5倍）-----
    proc.brightness(1.5)
    bright = proc.get_image()
    save_image(bright, "images/test_bright.jpg")
    proc.reset()

    # ---- 测试3：灰度 + 二值化 -----
    proc.to_gray().binarize(threshold=100)
    binary = proc.get_image()
    save_image(binary, "images/test_binary.jpg")
    proc.reset()

    # ---- 测试4：灰度 + 对比度增强（factor=2）-----
    proc.to_gray().contrast(2.0)
    high_contrast = proc.get_image()
    save_image(high_contrast, "images/test_contrast.jpg")
    proc.reset()

    # ---- 可视化对比 ----
    plt.figure(figsize=(15, 8))

    # 原图
    plt.subplot(2, 3, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    # 反转
    plt.subplot(2, 3, 2)
    plt.imshow(inverted)
    plt.title("Inverted")
    plt.axis("off")

    # 变亮
    plt.subplot(2, 3, 3)
    plt.imshow(bright)
    plt.title("Brightness x1.5")
    plt.axis("off")

    # 二值化（灰度显示）
    plt.subplot(2, 3, 4)
    plt.imshow(binary, cmap="gray")
    plt.title("Binary (th=100)")
    plt.axis("off")

    # 高对比度（灰度显示）
    plt.subplot(2, 3, 5)
    plt.imshow(high_contrast, cmap="gray")
    plt.title("Contrast x2.0")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()