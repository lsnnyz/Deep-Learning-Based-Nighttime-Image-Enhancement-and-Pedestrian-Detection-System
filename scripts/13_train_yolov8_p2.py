from ultralytics import YOLO


def main():
    print("🚀 正在初始化加入 P2 微小目标检测层的 YOLOv8s 模型...")

    # 核心变化1：从我们刚才写的 yaml 文件初始化全新的网络架构
    model = YOLO('yolov8s-p2.yaml')

    # 核心变化2：加载官方的 yolov8s.pt 权重作为预训练，加速收敛
    model.load('yolov8s.pt')

    print("🏃‍♂️ 开始在原图基准数据集上训练 P2 架构模型...")
    # 这里的 data 路径请确认是你当时 Baseline 数据集的 .yaml 路径
    results = model.train(
        data='llvip.yaml',  # 请根据你的实际路径调整
        epochs=50,  # 保持绝对公平的 50 轮
        batch=64,  # 保持公平的 batch 64
        imgsz=640,
        device=0,  # 使用 RTX 5090
        project='runs/detect',
        name='yolov8s_p2_baseline',  # 新模型的结果存放处
        workers=8,
        optimizer='auto'
    )

    print("🎉 YOLOv8s-P2 架构消融实验训练完成！")


if __name__ == '__main__':
    main()