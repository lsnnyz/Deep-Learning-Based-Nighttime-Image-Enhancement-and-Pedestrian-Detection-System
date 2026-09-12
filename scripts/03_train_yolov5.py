from ultralytics import YOLO


def main():
    # 只需要把 8 改成 5，框架会自动为你下载 YOLOv5s 的权重 (通常是适配新架构的 yolov5su.pt)
    model = YOLO('yolov5s.pt')

    print("开始训练 YOLOv5s 对比模型...")
    results = model.train(
        data='llvip.yaml',
        epochs=50,
        imgsz=640,
        # ⚠️ 核心：这里必须保持和 YOLOv8 完全一样的 batch size (64)，以保证实验严谨性
        batch=64,
        name='compare_yolov5s',  # 换个名字，防止覆盖了之前 YOLOv8 的结果
        device=0,
        workers=8
    )

    print("YOLOv5s 训练结束！模型保存在 runs/detect/compare_yolov5s/weights/best.pt")


if __name__ == '__main__':
    main()