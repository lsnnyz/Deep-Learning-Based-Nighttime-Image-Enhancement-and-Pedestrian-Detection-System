import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ================= 1. Zero-DCE 网络架构定义 =================
class DCE_Net(nn.Module):
    def __init__(self):
        super(DCE_Net, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        # 7层精简卷积，提取光照特征
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        # 输出 24 个通道的曲线参数估计图
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        # 迭代应用曲线进行提亮
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        x = x + r4 * (torch.pow(x, 2) - x)
        x = x + r5 * (torch.pow(x, 2) - x)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        return enhance_image


# ================= 2. 推断核心代码 =================
def enhance_image_zerodce(image_path, model, device):
    """读取图片并喂入网络进行增强"""
    # 读取并转为 RGB
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 转换为网络需要的 Tensor 格式: (Batch, Channel, Height, Width) 并归一化
    img_tensor = (np.asarray(img_rgb) / 255.0)
    img_tensor = torch.from_numpy(img_tensor).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # 闭包计算，不传递梯度
    with torch.no_grad():
        enhanced_tensor = model(img_tensor)

    # 将张量转换回 numpy 图像，并限制在 0-255 之间
    enhanced_img = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    enhanced_img = np.clip(enhanced_img * 255.0, 0, 255.0).astype(np.uint8)

    return img_rgb, enhanced_img


# ================= 3. 主函数与可视化 =================
def main():
    # 检测 5090 显卡
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Zero-DCE 正在使用设备: {device}")

    # 实例化网络并加载权重
    model = DCE_Net().to(device)
    model.load_state_dict(torch.load('Epoch99.pth', map_location=device, weights_only=True))
    model.eval()

    sample_dir = 'test_samples'
    output_dir = 'test_results'

    # 只拿你最关注的那张 img_3.png 来做专门的 1v1 对比
    target_img = 'img_3.png'
    img_path = os.path.join(sample_dir, target_img)

    if not os.path.exists(img_path):
        print(f"找不到文件 {img_path}，请确保该图片存在！")
        return

    print(f"正在使用 Zero-DCE 处理 {target_img} ...")
    original, enhanced = enhance_image_zerodce(img_path, model, device)

    # 绘制高级对比图
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    fig.suptitle('Deep Learning vs Darkness: Zero-DCE Enhancement', fontsize=16, fontweight='bold')

    axs[0].imshow(original)
    axs[0].set_title('Original Image (原图)')
    axs[0].axis('off')

    axs[1].imshow(enhanced)
    axs[1].set_title('Zero-DCE Output')
    axs[1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'zero_dce_{target_img}')
    plt.savefig(save_path)
    print(f"Zero-DCE 处理完成！结果已保存至: {save_path}")


if __name__ == '__main__':
    main()