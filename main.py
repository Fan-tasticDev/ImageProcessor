import argparse
import os
import sys
from utils.io import load_image, save_image
from utils.processor import ImageProcessor
from utils.batch import batch_process, OPERATIONS_MAP

# 将参数字符串解析为 batch_process 所需的字典格式
def parse_params(params_str: str) -> dict:
    """
    将参数字符串解析为字典。
    格式: "op1:key1=val1,key2=val2;op2:key3=val3"
    """
    if not params_str:
        return {}
    params_dict = {}
    # 按分号分割不同操作
    op_blocks = params_str.split(";")
    for block in op_blocks:
        block = block.strip()
        if not block:
            continue
        if ":" not in block:
            print(f"警告：参数格式错误，跳过 '{block}'")
            continue
        op_name, kv_pairs = block.split(":", 1)
        op_name = op_name.strip()
        kv_dict = {}
        # 按逗号分割 key=value
        for kv in kv_pairs.split(","):
            kv = kv.strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                # 尝试转换为数字
                try:
                    v = float(v)
                    if v.is_integer():
                        v = int(v)
                except ValueError:
                    pass  # 保持字符串
                kv_dict[k] = v
            else:
                # 只有键没有值，视为开关参数，设为True
                kv_dict[kv] = True
        params_dict[op_name] = kv_dict
    return params_dict

# 从 batch_process 中抽取，专门处理单张图片
def process_single_file(input_path: str, output_path: str, operations: list, params: dict):
    """处理单张图片并保存"""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")
    
    img = load_image(input_path)
    proc = ImageProcessor(img)
    
    for op_name in operations:
        if op_name not in OPERATIONS_MAP:
            print(f"警告: 未知操作 '{op_name}'，跳过")
            continue
        method = getattr(proc, OPERATIONS_MAP[op_name])
        kwargs = params.get(op_name, {})
        method(**kwargs)
    
    result = proc.get_image()
    save_image(result, output_path)
    print(f"处理完成，保存至: {output_path}")

# 根据 --input 的类型自动选择单张或批量模式
# --input / -i：输入图片路径或文件夹路径（必填）

# --output / -o：输出路径（若为文件夹则自动生成文件名，批量模式下为目录）

# --operation / -op：逗号分隔的操作名称列表，如 gray,invert,blur

# --params / -p：（可选）操作参数，格式为 "操作名:key=value,key=value;操作名:key=value"，用分号分隔不同操作，冒号后跟该操作的参数字典
def main():
    parser = argparse.ArgumentParser(
        description="图像处理命令行工具 - 支持单张和批量处理，操作：gray/invert/binarize/brightness/contrast/blur/sharpen/canny/equalize"
    )
    parser.add_argument("-i", "--input", required=True, help="输入图片路径或文件夹路径")
    parser.add_argument("-o", "--output", required=True, help="输出图片路径（单张）或输出文件夹（批量）")
    parser.add_argument("-op", "--operation", required=True,
                        help="操作列表，逗号分隔，如 gray,sharpen")
    parser.add_argument("-p", "--params", default="",
                        help='操作参数，格式: "操作名:key=val,key=val;操作名:key=val"，如 "binarize:threshold=100;sharpen:strength=1.5"')

    args = parser.parse_args()

    # 解析操作列表
    operations = [op.strip() for op in args.operation.split(",") if op.strip()]
    
    # 解析参数
    params = parse_params(args.params)

    # 判断输入是文件还是文件夹
    if os.path.isfile(args.input):
        # 单张图片处理
        output_path = args.output
        # 如果输出是目录，则在目录下生成文件名
        if os.path.isdir(output_path):
            # 自动使用原文件名+_processed后缀
            base_name = os.path.basename(args.input)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(output_path, f"{name}_processed{ext}")
        process_single_file(args.input, output_path, operations, params)
    
    elif os.path.isdir(args.input):
        # 批量处理
        batch_process(
            input_dir=args.input,
            output_dir=args.output,
            operations=operations,
            params=params
        )
        print(f"批量处理完成，结果保存在: {args.output}")
    else:
        print(f"错误：输入路径无效 '{args.input}'")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序运行出错: {e}")
        sys.exit(1)