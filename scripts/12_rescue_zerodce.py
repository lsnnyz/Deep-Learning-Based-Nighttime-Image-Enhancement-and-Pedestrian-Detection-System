from ultralytics import YOLO


def main():
    print("🚑 正在启动 Zero-DCE 专属抢救程序...")

    # 准确指向刚才中断的那个 last.pt
    model = YOLO('runs/detect/yolov8s_zerodce/weights/last.pt')

    # resume=True 是魔法！它会自动接上第 36 轮，跑完最后的 15 轮
    model.train(resume=True)

    print("🎉 Zero-DCE 抢救成功！所有增强模型全部炼丹完毕！")


if __name__ == '__main__':
    main()