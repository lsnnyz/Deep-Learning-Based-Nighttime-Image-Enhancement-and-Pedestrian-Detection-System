from ultralytics import YOLO


def main():
    # 1. 加载官方的 YOLOv8s 预训练模型 (会自动下载 yolov8s.pt)
    # 毕设选用 's' (small) 版本即可，参数量适中，方便跑通
    model = YOLO('yolov8s.pt')

    # 2. 开始训练
    # 注意：如果你的电脑没有 Nvidia 独立显卡，这里会自动用 CPU 跑，会非常慢。
    # epochs=50 意思是训练50轮，毕设通常50-100轮即可。
    # imgsz=640 是图片输入尺寸。
    print("开始训练 Baseline 模型...")
    results = model.train(
        data='llvip.yaml',
        epochs=50,  # 训练轮数
        imgsz=640,  # 图像大小
        batch=64,  # 批次大小 (如果显存不够报OOM，改成8或4)
        name='baseline_yolov8s',  # 本次训练结果保存的文件夹名字
        device=0  # 如果有显卡填0，全是CPU会忽略
    )

    print("训练结束！模型保存在 runs/detect/baseline_yolov8s/weights/best.pt")


if __name__ == '__main__':
    # 在 Windows 下运行多进程训练必须加 if __name__ == '__main__': 保护
    main()