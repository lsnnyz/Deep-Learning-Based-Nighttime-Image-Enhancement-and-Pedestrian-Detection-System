import matplotlib.pyplot as plt
import numpy as np


def main():
    # ================= 1. 填入你实验测得的真实数据 =================
    # 模型名称
    models = ['YOLOv8s', 'YOLOv5s', 'Faster R-CNN']

    # Y轴：精度 mAP@50 (%)
    mAP = [89.10, 89.10, 88.09]

    # X轴：速度 FPS (YOLO的FPS可由 1000/(预处理+推断+后处理ms) 估算，这里用估算值)
    fps = [1250, 1300, 93.4]

    # 气泡大小：参数量 Parameters (单位: M)
    # 大小会决定图上圆圈的面积
    params = [11.1, 9.1, 41.5]

    # ================= 2. 论文级图表样式设置 =================
    # 尽可能使用经典的 Times New Roman 字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # 创建画布 (宽高比 8:6，清晰度 300 dpi 满足打印出版要求)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 设置背景网格 (虚线，透明度0.5，显得专业干净)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

    # 定义三种高级莫兰迪配色 (区分三个模型)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    # ================= 3. 绘制气泡图 =================
    # 根据参数量放大圆圈面积，方便视觉观察 (这里乘以 30 是一个缩放系数)
    bubble_sizes = [p * 30 for p in params]

    scatter = ax.scatter(fps, mAP, s=bubble_sizes, c=colors,
                         alpha=0.7, edgecolors='white', linewidth=2, zorder=3)

    # ================= 4. 添加数据标签 =================
    for i, model in enumerate(models):
        # 在圆圈旁边写上模型名字和具体参数量
        label_text = f"{model}\n({params[i]}M)"
        # 为了防止文字重叠，略微偏移坐标
        ax.text(fps[i] + 30, mAP[i] - 0.05, label_text,
                fontsize=11, fontweight='bold', color='#333333',
                verticalalignment='top')

    # ================= 5. 修饰坐标轴和标题 =================
    ax.set_title('Performance Comparison of Detection Models on LLVIP Dataset',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Inference Speed (FPS) $\\rightarrow$ Better', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy mAP@50 (%) $\\rightarrow$ Better', fontsize=12, fontweight='bold')

    # 设置坐标轴留白，防止气泡被切掉
    ax.set_xlim(0, 1500)
    ax.set_ylim(87.0, 90.0)

    # 绘制一个“理想区域”指示箭头 (右上角是最好的)
    ax.annotate('Ideal Region\n(Fast & Accurate)', xy=(1400, 89.8), xytext=(1100, 89.5),
                arrowprops=dict(facecolor='gray', shrink=0.05, width=2, headwidth=8),
                fontsize=11, color='gray', style='italic')

    # ================= 6. 保存与显示 =================
    # 自动紧凑布局
    plt.tight_layout()

    # 保存为高清图片和矢量图 (矢量图放到Word/LaTeX里放大不失真！)
    plt.savefig('model_comparison_bubble.png', dpi=300)
    plt.savefig('model_comparison_bubble.pdf')
    print("绘图成功！图片已保存为 model_comparison_bubble.png 和 pdf")


if __name__ == '__main__':
    main()