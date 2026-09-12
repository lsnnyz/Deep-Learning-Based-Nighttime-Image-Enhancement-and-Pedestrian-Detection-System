import os
import cv2
import time
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

# ================= 配置路径 =================
VAL_IMG_DIR = '../datasets/LLVIP/images/val'
VAL_LABEL_DIR = '../datasets/LLVIP/labels/val'
WEIGHTS_PATH = 'faster_rcnn_llvip.pth'


# ================= 1. 自定义数据集类 (验证集) =================
class LLVIP_FasterRCNN_Dataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.imgs = [img for img in os.listdir(img_dir) if img.endswith('.jpg')]

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        _, h, w = img_tensor.shape

        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, x_center, y_center, width, height = map(float, parts)
                        xmin = (x_center - width / 2) * w
                        ymin = (y_center - height / 2) * h
                        xmax = (x_center + width / 2) * w
                        ymax = (y_center + height / 2) * h
                        xmin, ymin = max(0, xmin), max(0, ymin)
                        xmax, ymax = min(w, xmax), min(h, ymax)
                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(1)

        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        return img_tensor, target

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    return tuple(zip(*batch))


# ================= 2. 加载模型与权重 =================
def load_model(weights_path, num_classes, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 加载你刚才训练好的权重
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()  # 必须设置为评估模式
    return model


# ================= 3. 评估主循环 =================
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"正在使用 {device} 进行推断评估...")

    dataset = LLVIP_FasterRCNN_Dataset(VAL_IMG_DIR, VAL_LABEL_DIR)
    # 评估时 batch_size 设小一点，防止预测框太多挤爆显存
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = load_model(WEIGHTS_PATH, num_classes=2, device=device)

    # 实例化 torchmetrics 的 mAP 计算器
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

    print("开始在验证集上进行预测...")
    total_time = 0
    total_images = 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")
        for images, targets in progress_bar:
            images = list(image.to(device) for image in images)

            # 记录推理时间 (用于对比 FPS)
            start_time = time.time()
            outputs = model(images)
            torch.cuda.synchronize()  # 确保 GPU 计算完成
            end_time = time.time()

            total_time += (end_time - start_time)
            total_images += len(images)

            # 将 target 移到设备上以供计算
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 更新 metric
            metric.update(outputs, targets)

    print("\n================ 最终成绩单 ================")
    print("正在计算 mAP，这可能需要一两分钟，请稍候...")
    results = metric.compute()

    print(f"mAP@50 (核心精度): {results['map_50'].item():.4f}")
    print(f"mAP@50-95 (框质量): {results['map'].item():.4f}")

    # 计算 FPS
    fps = total_images / total_time
    print(f"推断速度 (FPS): {fps:.2f} 张/秒")
    print("===========================================")


if __name__ == '__main__':
    main()