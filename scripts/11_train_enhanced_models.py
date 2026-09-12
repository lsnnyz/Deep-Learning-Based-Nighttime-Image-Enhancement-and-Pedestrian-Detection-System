from ultralytics import YOLO


def main():
    # 定义要排队训练的数据集和对应的保存文件夹名称
    # 格式: ('yaml配置文件名', '训练结果保存的名字')
    tasks = [
        ('llvip_gamma.yaml', 'yolov8s_gamma'),
        ('llvip_clahe.yaml', 'yolov8s_clahe'),
        ('llvip_retinex.yaml', 'yolov8s_retinex'),
        ('llvip_zerodce.yaml', 'yolov8s_zerodce')
    ]

    print("==================================================")
    print("🚀 开启 YOLOv8 增强数据集批量训练流水线")
    print("==================================================")

    for yaml_file, run_name in tasks:
        print(f"\n---> 当前正在准备训练任务: {run_name}")
        print(f"---> 使用数据集配置文件: {yaml_file}")

        # 【核心控制变量1】：每次都必须重新加载最原始的 yolov8s.pt
        # 绝对不能用上一次训练过的权重接着练，必须保证大家的“起跑线”完全一样！
        model = YOLO('yolov8s.pt')

        # 【核心控制变量2】：保持与 Baseline 完全一致的超参数
        model.train(
            data=yaml_file,
            epochs=50,  # 与 Baseline 一致
            imgsz=640,  # 与 Baseline 一致
            batch=64,  # 完美喂饱你的 5090
            name=run_name,  # 结果将分别保存在 runs/detect/yolov8s_gamma 等文件夹下
            device=0,  # 指定使用你的 5090 显卡
            workers=8  # 数据加载线程数
        )

        print(f"✅ 任务 {run_name} 训练完成！最强权重已保存在 runs/detect/{run_name}/weights/best.pt")

    print("\n🎉🎉🎉 恭喜！所有 4 个增强模型的训练任务已全部圆满结束！")
    print("你可以去 runs/detect/ 目录下查看它们的 results.png 来对比 mAP 了！")


if __name__ == '__main__':
    main()