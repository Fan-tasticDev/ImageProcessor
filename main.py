import matplotlib.pyplot as plt
from utils.io import load_image, save_image
from utils.transforms import rgb_to_gray

def main():
    # 1. 加载图片
    img_path = "images/test.jpg"   # 改成你的图片名
    original = load_image(img_path)
    print(f"原图尺寸: {original.shape}")

    # 2. 转灰度
    gray = rgb_to_gray(original)
    print(f"灰度图尺寸: {gray.shape}")

    # 3. 保存灰度图
    output_path = "images/test_gray.jpg"
    save_image(gray, output_path)

    # 4. 用 Matplotlib 对比显示
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original RGB")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(gray, cmap="gray")   # 灰度图必须指定 cmap
    plt.title("Grayscale")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()