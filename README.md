# Image Processor - Python 图像处理命令行工具

一个基于 Python 的图像处理系统，支持灰度转换、反转、二值化、亮度/对比度调整、高斯模糊、锐化、Canny 边缘检测、直方图均衡化，以及批量处理功能。

## 功能特性
- 🖼️ **基础变换**：灰度化、反转、二值化、亮度/对比度调整
- 🎨 **滤波增强**：高斯模糊、锐化滤镜
- 📐 **边缘检测**：Canny 边缘检测（简化版）
- 📊 **直方图均衡化**：自动拉伸图像对比度
- 📁 **批量处理**：一键处理整个文件夹的图片
- ⌨️ **命令行工具**：通过 argparse 提供的友好 CLI 接口
- 🔗 **链式调用**：ImageProcessor 类支持操作链，代码更简洁

## 环境要求
- Python 3.9+
- 依赖库（见 `requirements.txt`）

## 安装与配置
```bash
# 克隆仓库
git clone https://github.com/Fan-tasticDev/ImageProcessor.git
cd ImageProcessor

# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

# 快速开始
## 1. 单张图片处理
```bash
python main.py -i images/test.jpg -o output/result.jpg -op gray,equalize
```
## 2. 批量处理
```bash
python main.py -i images -o output/batch -op gray,sharpen
```
## 3. 带参数操作
```bash
python main.py -i images/test.jpg -o output/canny.jpg -op gray,canny -p "canny:low=50,high=150"
```

# 查看完整帮助
```bash
python main.py -h
```

# 支持的操作
| 操作名 | 说明 | 可选参数 |
| ---- | ---- | ---- |
| gray | 灰度化 | 无 |
| invert | 颜色反转 | 无 |
| binarize | 二值化 | threshold (int, 默认128) |
| brightness | 亮度调整 | factor (float, 默认1.0) |
| contrast | 对比度调整 | factor (float, 默认1.0) |
| blur | 高斯模糊 | kernel_size (int), sigma (float) |
| sharpen | 锐化 | strength (float, 默认1.0) |
| canny | Canny边缘检测 | low (float), high (float) |
| equalize | 直方图均衡化 | 无 |

# 项目结构
```text
ImageProcessor/
├── main.py                # 命令行入口
├── utils/
│   ├── io.py              # 图像读写
│   ├── transforms.py      # 图像变换函数
│   ├── processor.py       # ImageProcessor 工具类
│   └── batch.py           # 批量处理逻辑
├── images/                # 示例图片
├── requirements.txt       # 依赖列表
└── README.md
```