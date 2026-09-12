import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

# ================= 配置路径 =================
# 添加 ../ 让程序先返回上一层目录
XML_DIR = '../LLVIPdata/Annotations'
IMG_TRAIN_DIR = '../LLVIPdata/visible/train'
IMG_TEST_DIR = '../LLVIPdata/visible/test'
OUTPUT_DIR = '../datasets/LLVIP'  # YOLO标准数据集也会生成在根目录的 datasets 下

# 类别定义 (LLVIP 主要是行人)
CLASSES = ["person"]


def convert_box(size, box):
    """将 XML 的绝对坐标转换为 YOLO 的归一化中心坐标"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def process_dataset(img_source_dir, split_name):
    print(f"正在处理 {split_name} 集...")
    # split_name 为 'train' 或 'val'
    output_img_dir = os.path.join(OUTPUT_DIR, 'images', split_name)
    output_txt_dir = os.path.join(OUTPUT_DIR, 'labels', split_name)

    # 创建对应的文件夹
    Path(output_img_dir).mkdir(parents=True, exist_ok=True)
    Path(output_txt_dir).mkdir(parents=True, exist_ok=True)

    # 获取当前目录下所有的图片文件
    img_files = [f for f in os.listdir(img_source_dir) if f.endswith('.jpg')]

    for img_file in img_files:
        # 寻找对应的 xml 标注文件
        xml_file = img_file.replace('.jpg', '.xml')
        xml_path = os.path.join(XML_DIR, xml_file)
        img_path = os.path.join(img_source_dir, img_file)
        txt_path = os.path.join(output_txt_dir, img_file.replace('.jpg', '.txt'))

        if not os.path.exists(xml_path):
            print(f"警告: 找不到对应的标注文件 {xml_path}，跳过该图片")
            continue

        # 解析 XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        has_person = False
        with open(txt_path, 'w') as out_file:
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                if cls_name not in CLASSES:
                    continue
                has_person = True
                cls_id = CLASSES.index(cls_name)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_box((w, h), b)
                out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")

        # 如果有行人目标，才将图片拷贝到 YOLO 目录
        if has_person:
            shutil.copy(img_path, os.path.join(output_img_dir, img_file))
        else:
            os.remove(txt_path)  # 删除空txt文件


def main():
    # 按照官方目录处理
    process_dataset(IMG_TRAIN_DIR, 'train')
    # 将官方的 test 作为 YOLO 的 val
    process_dataset(IMG_TEST_DIR, 'val')

    print("数据集转换完成！请检查 datasets/LLVIP 文件夹。")


if __name__ == '__main__':
    main()