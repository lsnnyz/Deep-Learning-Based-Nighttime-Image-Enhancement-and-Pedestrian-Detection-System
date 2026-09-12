import argparse
import gc
import os

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch training for YOLOv8s-P2 cross experiments."
    )
    parser.add_argument(
        "--shutdown-after-finish",
        action="store_true",
        help="Shut down the machine after all experiments finish. Disabled by default for safety.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 定义所有交叉实验配置
    # 格式：('数据集 yaml', '结果保存目录名')
    experiments = [
        ('llvip_gamma.yaml', 'yolov8s_p2_gamma'),
        ('llvip_clahe.yaml', 'yolov8s_p2_clahe'),
        ('llvip_retinex.yaml', 'yolov8s_p2_retinex'),
        ('llvip_zerodce.yaml', 'yolov8s_p2_zerodce'),
    ]

    print("=" * 60)
    print("Starting YOLOv8s-P2 cross-experiment training pipeline")
    print(f"Total experiments: {len(experiments)}")
    print("=" * 60)

    for i, (data_yaml, run_name) in enumerate(experiments, start=1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(experiments)}] Running: {run_name}")
        print(f"Dataset config: {data_yaml}")
        print(f"{'=' * 60}\n")

        # 每组实验都从相同的 YOLOv8s-P2 结构和官方预训练权重开始，
        # 避免继承上一组实验的训练结果，保证横向对比公平。
        model = YOLO('yolov8s-p2.yaml')
        model.load('yolov8s.pt')

        model.train(
            data=data_yaml,
            epochs=50,
            batch=64,
            imgsz=640,
            device=0,
            project='runs/detect',
            name=run_name,
            workers=4,
            optimizer='auto',
        )

        print(
            f"Finished {run_name}. "
            f"Best weights: runs/detect/{run_name}/weights/best.pt"
        )

        # 清理显存和 Python 对象，降低连续训练多组实验时的资源占用。
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll cross experiments have finished.")

    # 安全设计：默认绝不自动关机。
    # 只有用户明确传入 --shutdown-after-finish 时才执行关机命令。
    if args.shutdown_after_finish:
        print("Shutdown requested by --shutdown-after-finish.")
        if os.name == 'nt':
            exit_code = os.system('shutdown /s /t 0')
        else:
            exit_code = os.system('shutdown -h now')

        if exit_code != 0:
            print(
                "Shutdown command failed. "
                "Please check system permissions or shut down manually."
            )
    else:
        print(
            "Automatic shutdown is disabled. "
            "Use --shutdown-after-finish only when you explicitly want it."
        )


if __name__ == '__main__':
    main()
