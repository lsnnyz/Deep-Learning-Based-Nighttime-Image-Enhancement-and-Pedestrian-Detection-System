import cv2
import numpy as np
import torch
import torch.nn as nn
import csv
from pathlib import Path


# ================= 1. 定义 Zero-DCE 网络 =================
class DCE_Net(nn.Module):
    def __init__(self):
        super(DCE_Net, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
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
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
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


# ================= 2. 图像增强算法实现 =================
def apply_gamma(img, gamma=0.45):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def apply_clahe(img, clip_limit=2.5, grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def apply_retinex(img, sigma=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = v.astype(np.float64) + 1.0
    blur = cv2.GaussianBlur(v, (0, 0), sigma)
    retinex = np.log10(v) - np.log10(blur)
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255
    hsv[:, :, 2] = np.clip(retinex, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_zerodce(img, model, device):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = (np.asarray(img_rgb) / 255.0)
    img_tensor = torch.from_numpy(img_tensor).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        enhanced_tensor = model(img_tensor)
    enhanced_img = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    enhanced_img = np.clip(enhanced_img * 255.0, 0, 255.0).astype(np.uint8)
    return cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)


# ================= 3. 客观指标计算公式 =================
def calc_entropy(img):
    """计算信息熵"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return entropy


def calc_avg_gradient(img):
    """计算平均梯度"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 计算相邻像素差分
    dx = np.diff(gray, axis=1)[:-1, :]
    dy = np.diff(gray, axis=0)[:, :-1]
    # 根据公式计算梯度矩阵并求均值
    grad = np.sqrt((dx ** 2 + dy ** 2) / 2.0)
    return np.mean(grad)


# ================= 4. 主函数：生成对比表格 =================
def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    weights_path = script_dir / 'Epoch99.pth'
    img_path = script_dir / 'test_samples' / 'img_3.png'
    output_dir = project_root / 'result'
    output_csv = output_dir / 'iqa_results.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DCE_Net().to(device)
    if not weights_path.exists():
        print(f"找不到模型权重 {weights_path}，请检查。")
        return
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    # 读取测试图片 (以 img_3.png 为例)
    if not img_path.exists():
        print(f"找不到图片 {img_path}，请检查。")
        return

    img_origin = cv2.imread(str(img_path))

    # 生成4种增强结果
    print("正在处理图像并计算客观指标，请稍候...")
    img_gamma = apply_gamma(img_origin.copy())
    img_clahe = apply_clahe(img_origin.copy())
    img_retinex = apply_retinex(img_origin.copy())
    img_zerodce = apply_zerodce(img_origin.copy(), model, device)

    # 汇总计算
    results = {
        "Baseline (原图)": img_origin,
        "Gamma 校正": img_gamma,
        "CLAHE 算法": img_clahe,
        "Retinex (SSR)": img_retinex,
        "Zero-DCE": img_zerodce
    }

    print("\n" + "=" * 50)
    print(f"{'算法名称':<15} | {'信息熵 (Entropy) ↑':<15} | {'平均梯度 (Avg Gradient) ↑'}")
    print("-" * 50)

    rows = []
    for name, img in results.items():
        entropy = calc_entropy(img)
        grad = calc_avg_gradient(img)
        rows.append([name, f"{entropy:.4f}", f"{grad:.4f}"])
        print(f"{name:<18} | {entropy:<18.4f} | {grad:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'entropy', 'avg_gradient'])
        writer.writerows(rows)

    print("=" * 50)
    print("注：↑ 表示数值越大越好。")
    print(f"IQA 结果已保存到: {output_csv}")


if __name__ == '__main__':
    main()
