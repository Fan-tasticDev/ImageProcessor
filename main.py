import matplotlib.pyplot as plt
from utils.io import load_image, save_image
from utils.processor import ImageProcessor

def main():
    img_path = "images/test.jpg"
    original = load_image(img_path)
    proc = ImageProcessor(original)

    # 先转为灰度，保存用于对比
    proc.to_gray()
    gray_original = proc.get_image()
    save_image(gray_original, "images/test_gray.jpg")
    
    # 直方图均衡化
    proc.equalize_hist()
    equalized = proc.get_image()
    save_image(equalized, "images/test_equalized.jpg")
    
    # ---- 1. 图片对比：原灰度图 vs 均衡化图 ----
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(gray_original, cmap="gray")
    plt.title("Original Gray")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(equalized, cmap="gray")
    plt.title("Histogram Equalized")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # ---- 2. 直方图对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].hist(gray_original.ravel(), bins=256, range=(0,255), color='gray', alpha=0.7)
    axes[0].set_title("Original Gray Histogram")
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Frequency")
    
    axes[1].hist(equalized.ravel(), bins=256, range=(0,255), color='black', alpha=0.7)
    axes[1].set_title("Equalized Histogram")
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

    # 模拟暗图：亮度设为原图的0.3倍
    proc_dark = ImageProcessor(original)
    proc_dark.brightness(0.3)
    dark_img = proc_dark.get_image()
    save_image(dark_img, "images/test_dark.jpg")

    # 对暗图做均衡化
    proc_dark.to_gray().equalize_hist()
    dark_eq = proc_dark.get_image()
    save_image(dark_eq, "images/test_dark_eq.jpg")

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(dark_img, cmap='gray')
    plt.title("Dark Image")
    plt.subplot(1,2,2)
    plt.imshow(dark_eq, cmap='gray')
    plt.title("After Equalization")
    plt.show()

if __name__ == "__main__":
    main()