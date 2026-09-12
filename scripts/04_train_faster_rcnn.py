import os
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ================= 配置路径 =================
# 复用我们之前生成好的 YOLO 格式数据集
TRAIN_IMG_DIR = '../datasets/LLVIP/images/train'
TRAIN_LABEL_DIR = '../datasets/LLVIP/labels/train'


# ================= 1. 自定义数据集类 =================
class LLVIP_FasterRCNN_Dataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.imgs = [img for img in os.listdir(img_dir) if img.endswith('.jpg')]

    def __getitem__(self, idx):
        # 1. 读取图像
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 图像转为 PyTorch 需要的格式 (C, H, W) 并归一化到 0-1
        img_tensor = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        _, h, w = img_tensor.shape

        # 2. 读取标签 (将 YOLO 的中心点归一化坐标，转为 Faster R-CNN 的绝对边界框 [xmin, ymin, xmax, ymax])
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)

                        xmin = (x_center - width / 2) * w
                        ymin = (y_center - height / 2) * h
                        xmax = (x_center + width / 2) * w
                        ymax = (y_center + height / 2) * h

                        # 防止坐标越界
                        xmin, ymin = max(0, xmin), max(0, ymin)
                        xmax, ymax = min(w, xmax), min(h, ymax)

                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(1)  # Faster R-CNN 中 0 是背景，1 是行人

        # 3. 构造 target 字典
        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        return img_tensor, target

    def __len__(self):
        return len(self.imgs)


# PyTorch 要求的自定义批次拼接函数
def collate_fn(batch):
    return tuple(zip(*batch))


# ================= 2. 获取并修改模型 =================
def get_model(num_classes):
    # 加载基于 ResNet50 的预训练 Faster R-CNN 模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # 获取分类器的输入特征维度
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 替换头部，适应我们的类别数 (行人 + 背景 = 2类)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ================= 3. 训练主循环 =================
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"正在使用设备: {device}")

    # 实例化数据集与加载器 (Faster R-CNN 极度吃显存，5090 建议 batch_size 设置为 8 或 16)
    dataset = LLVIP_FasterRCNN_Dataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # 实例化模型 (2类: 行人 + 背景)
    model = get_model(num_classes=2)
    model.to(device)

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 10  # Faster R-CNN 训练较慢，先跑 10 轮看看

    print("开始训练 Faster R-CNN...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # 进度条
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for images, targets in progress_bar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 前向传播，Faster R-CNN 在 train 模式下直接返回 loss 字典
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            progress_bar.set_postfix(loss=losses.item())

        print(f"Epoch {epoch + 1} 平均 Loss: {total_loss / len(data_loader):.4f}")

    # 保存权重
    torch.save(model.state_dict(), 'faster_rcnn_llvip.pth')
    print("训练结束！权重已保存至 faster_rcnn_llvip.pth")


if __name__ == '__main__':
    main()