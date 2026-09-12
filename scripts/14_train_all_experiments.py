from ultralytics import YOLO
import gc
import torch
import os


def main():
    # 定义你要跑的所有实验配置矩阵
    # 格式：('你的数据集yaml名字', '保存结果的文件夹名字')
    experiments = [
        ('llvip_gamma.yaml', 'yolov8s_p2_gamma'),
        ('llvip_clahe.yaml', 'yolov8s_p2_clahe'),
        ('llvip_retinex.yaml', 'yolov8s_p2_retinex'),
        ('llvip_zerodce.yaml', 'yolov8s_p2_zerodce')
    ]

    print("🚀 正在启动暗光行人检测 P2 架构大满贯交叉实验流水线...")
    print(f"📋 共计 {len(experiments)} 个实验任务待执行，预计总耗时约 8 小时。")

    for i, (data_yaml, run_name) in enumerate(experiments):
        print(f"\n{'=' * 60}")
        print(f"🔥 [任务 {i + 1}/{len(experiments)}] 正在执行: {run_name}")
        print(f"📁 使用数据集配置: {data_yaml}")
        print(f"{'=' * 60}\n")

        # 【学术严谨性极度重要】:
        # 每次实验都必须从你写的 yolov8s-p2.yaml 重新初始化一个干净的网络
        # 绝对不能用上一次跑完的权重接着跑，否则就不是公平对比了！
        model = YOLO('yolov8s-p2.yaml')
        model.load('yolov8s.pt')  # 加载官方基础权重

        # 启动训练 (保持求稳的配置)
        model.train(
            data=data_yaml,
            epochs=50,
            batch=64,  # 稳妥不爆显存
            imgsz=640,
            device=0,  # RTX 5090
            project='runs/detect',
            name=run_name,
            workers=4,
            optimizer='auto'
        )

        print(f"✅ 实验 {run_name} 执行完毕！权重已保存在 runs/detect/{run_name}/weights/best.pt")

        # 【防显存泄漏清理】: 跑完一个模型后，清空一下 GPU 显存缓存，干干净净迎接下一个模型
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n🎉🎉🎉 所有 4 个交叉实验已全部自动执行完毕！你的核心科研工作量闭环了！")

    # 【省钱大法】：所有实验跑完后，向服务器发送关机指令
    print("⏳ 正在安全关闭 AutoDL 服务器...")
    os.system("shutdown")


if __name__ == '__main__':
    main()