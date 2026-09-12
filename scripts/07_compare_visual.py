import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ================= 1. 算法实现 =================

def apply_gamma(img, gamma=0.45):
    """Gamma校正: 调整 gamma 值 (必须小于1，越小越亮)"""
    # 修复Bug：直接使用 gamma 作为指数
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def apply_clahe(img, clip_limit=2.5, grid_size=(8, 8)):
    """CLAHE: 调整 clip_limit (越大越亮，但噪点越爆炸)"""
    # 转换到 LAB 颜色空间，只对亮度通道 L 进行处理，避免色彩失真
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def apply_retinex(img, sigma=50):
    """单尺度 Retinex (SSR): 调整 sigma (高斯模糊尺度)"""
    # 转换到 HSV 空间，只处理 V 通道
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = v.astype(np.float64) + 1.0  # 加1防止log(0)

    blur = cv2.GaussianBlur(v, (0, 0), sigma)
    retinex = np.log10(v) - np.log10(blur)

    # 线性拉伸回 0-255
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255
    hsv[:, :, 2] = np.clip(retinex, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ================= 2. 批量处理与绘图 =================

def main():
    sample_dir = 'test_samples'
    output_dir = 'test_results'
    os.makedirs(output_dir, exist_ok=True)

    # 获取测试文件夹下的所有图片
    image_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        print("请先挑选几张 .jpg 图片放进 test_samples 文件夹中！")
        return

    # 设置matplotlib支持中文字体(可选)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    for img_name in image_files:
        img_path = os.path.join(sample_dir, img_name)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # matplotlib 需要 RGB 格式

        # 运行算法 (在这里修改参数进行测试！)
        gamma_img = apply_gamma(img_bgr, gamma=0.45)
        clahe_img = apply_clahe(img_bgr, clip_limit=2.5)
        retinex_img = apply_retinex(img_bgr, sigma=50)

        # 转换为 RGB 供显示
        gamma_rgb = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB)
        clahe_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)
        retinex_rgb = cv2.cvtColor(retinex_img, cv2.COLOR_BGR2RGB)

        # 绘制 2x2 四宫格对比图
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=200)
        fig.suptitle(f'图像增强算法视觉对比 - {img_name}', fontsize=16, fontweight='bold')

        axs[0, 0].imshow(img_rgb)
        axs[0, 0].set_title('Original (原图)')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(gamma_rgb)
        axs[0, 1].set_title('Gamma ($\gamma=0.45$)')
        axs[0, 1].axis('off')

        axs[1, 0].imshow(clahe_rgb)
        axs[1, 0].set_title('CLAHE (clipLimit=2.5)')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(retinex_rgb)
        axs[1, 1].set_title('Retinex ($\sigma=50$)')
        axs[1, 1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'compare_{img_name}')
        plt.savefig(save_path)
        plt.close()
        print(f"已生成对比图: {save_path}")


if __name__ == '__main__':
    main()