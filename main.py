import matplotlib.pyplot as plt
from utils.io import load_image, save_image
from utils.processor import ImageProcessor


def main():
    img_path = "images/test.jpg"
    original = load_image(img_path)
    proc = ImageProcessor(original)

    # ---- 测试高斯模糊 ----
    proc.blur(kernel_size=5, sigma=2)
    blurred = proc.get_image()
    save_image(blurred, "images/test_blur.jpg")
    proc.reset()

    # ---- 测试锐化 ----
    proc.sharpen(strength=1.5)
    sharpened = proc.get_image()
    save_image(sharpened, "images/test_sharp.jpg")
    proc.reset()

    # ---- 测试Canny边缘检测 ----
    proc.canny(low=50, high=150)
    edges = proc.get_image()
    save_image(edges, "images/test_canny.jpg")
    proc.reset()

    # ---- 可视化 ----
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(blurred)
    plt.title("Gaussian Blur sigma=2")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(sharpened)
    plt.title("Sharpen strength=1.5")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(edges, cmap="gray")
    plt.title("Canny Edges (50,150)")
    plt.axis("off")

    # 额外显示：原图转灰度后与边缘叠加
    plt.subplot(2, 3, 5)
    gray = proc.to_gray().get_image()
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale for comparison")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # 实验：不同sigma的高斯模糊


    proc2 = ImageProcessor(original)
    plt.figure(figsize=(10, 4))
    for i, s in enumerate([0.5, 2, 4]):
        proc2.reset()
        proc2.blur(sigma=s)
        plt.subplot(1, 3, i + 1)
        plt.imshow(proc2.get_image())
        plt.title(f"sigma={s}")
        plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
