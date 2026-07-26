from utils.batch import batch_process

if __name__ == "__main__":
    # 批量处理：灰度 → 均衡化 → 锐化
    batch_process(
        input_dir="images",
        output_dir="images_output/batch_test",
        operations=["gray", "equalize", "sharpen"],
        params={"sharpen": {"strength": 1.2}}
    )

    # 组合1：只做Canny边缘检测
    batch_process("images", "images_output/canny", ["gray", "canny"], 
              params={"canny": {"low": 50, "high": 100}})

    # 组合2：变亮 + 反转
    batch_process("images", "images_output/invert_bright", ["brightness", "invert"],
              params={"brightness": {"factor": 1.5}})
    print("批量处理完成！")