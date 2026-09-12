import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import shutil
from tqdm import tqdm


# ================= 1. Zero-DCE 网络定义 =================
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


# ================= 2. 算法实现 =================
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


# ================= 3. 核心处理流程 =================
def process_dataset():
    # 基础路径配置
    source_base_dir = '../datasets/LLVIP'  # 你原始生成 YOLO 格式的数据集路径

    # 检查原始数据集是否存在
    if not os.path.exists(source_base_dir):
        print(f"错误: 找不到原始数据集 {source_base_dir}")
        return

    # 定义要生成的 4 个新数据集的后缀和对应的处理函数
    algorithms = {
        'Gamma': apply_gamma,
        'CLAHE': apply_clahe,
        'Retinex': apply_retinex,
        'ZeroDCE': 'zero_dce_special'  # 特殊标记，需要传模型
    }

    # 加载 Zero-DCE 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DCE_Net().to(device)
    model.load_state_dict(torch.load('Epoch99.pth', map_location=device, weights_only=True))
    model.eval()

    splits = ['train', 'val']

    for algo_name, func in algorithms.items():
        target_base_dir = f'../datasets/LLVIP_{algo_name}'
        print(f"\n================ 开始生成 {algo_name} 数据集 ================")

        for split in splits:
            src_img_dir = os.path.join(source_base_dir, 'images', split)
            src_lbl_dir = os.path.join(source_base_dir, 'labels', split)

            tgt_img_dir = os.path.join(target_base_dir, 'images', split)
            tgt_lbl_dir = os.path.join(target_base_dir, 'labels', split)

            os.makedirs(tgt_img_dir, exist_ok=True)
            os.makedirs(tgt_lbl_dir, exist_ok=True)

            # 1. 直接拷贝对应的 labels 文件夹（无需修改标签内容）
            print(f"正在拷贝 {split} 标签...")
            for lbl_file in os.listdir(src_lbl_dir):
                shutil.copy2(os.path.join(src_lbl_dir, lbl_file), os.path.join(tgt_lbl_dir, lbl_file))

            # 2. 处理图像
            img_files = [f for f in os.listdir(src_img_dir) if f.endswith(('.jpg', '.png'))]
            print(f"正在增强 {split} 图像 (共 {len(img_files)} 张)...")

            for img_file in tqdm(img_files, desc=f"{algo_name}-{split}"):
                img_path = os.path.join(src_img_dir, img_file)
                save_path = os.path.join(tgt_img_dir, img_file)

                # 如果图片已经存在，则跳过（方便断点续传）
                if os.path.exists(save_path):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue

                # 应用对应的算法
                if algo_name == 'ZeroDCE':
                    enhanced_img = apply_zerodce(img, model, device)
                else:
                    enhanced_img = func(img)

                cv2.imwrite(save_path, enhanced_img)

        # 3. 为每个数据集自动生成对应的 yaml 配置文件
        yaml_content = f"""path: ../datasets/LLVIP_{algo_name}
train: images/train
val: images/val
names:
  0: pedestrian
"""
        with open(f'llvip_{algo_name.lower()}.yaml', 'w') as f:
            f.write(yaml_content)

        print(f"[{algo_name}] 数据集处理完成！已生成配置文件 llvip_{algo_name.lower()}.yaml")


if __name__ == '__main__':
    process_dataset()