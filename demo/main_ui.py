# -*- coding: utf-8 -*-
"""
夜间图像增强与行人检测可视化系统

本程序基于 PyQt5 构建图形用户界面，结合 OpenCV 完成夜间图像增强，
并调用 Ultralytics YOLO 模型对图片或视频中的行人目标进行检测。
主要功能包括：
1. 加载训练得到的 YOLO 检测模型权重；
2. 读取待检测图片或视频文件；
3. 支持原图、Gamma 校正、CLAHE、Retinex 和 Zero-DCE 近似增强；
4. 显示原始输入与增强检测后的可视化结果；
5. 保存当前检测结果图像。
"""

import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO


class NightPedestrianDetectionUI(QMainWindow):
    """夜间图像增强与行人检测系统主窗口。"""

    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 750
    DISPLAY_WIDTH = 480
    DISPLAY_HEIGHT = 360
    VIDEO_TIMER_INTERVAL_MS = 30
    PERSON_CLASS_ID = 0

    def __init__(self) -> None:
        super().__init__()

        self.model: Optional[YOLO] = None
        self.media_path: Optional[str] = None
        self.media_type: Optional[str] = None
        self.cv_image: Optional[np.ndarray] = None
        self.annotated_image: Optional[np.ndarray] = None
        self.is_playing = False
        self.cap: Optional[cv2.VideoCapture] = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_next_frame)

        self.init_ui()

    def init_ui(self) -> None:
        """初始化窗口控件和页面布局。"""
        self.setWindowTitle("夜间图像增强与行人检测系统")
        self.resize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #f0f0f0;")
        self.setCentralWidget(central_widget)

        title_label = QLabel("夜间图像增强与行人检测可视化系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        title_label.setStyleSheet("margin: 10px; color: black;")

        control_layout = self.build_control_panel()
        display_layout = self.build_display_panel()

        left_widget = QWidget()
        left_widget.setLayout(control_layout)
        left_widget.setFixedWidth(250)

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        main_layout.addLayout(display_layout)

        final_layout = QVBoxLayout()
        final_layout.addWidget(title_label)
        final_layout.addLayout(main_layout)

        central_widget.setLayout(final_layout)

    def build_control_panel(self) -> QVBoxLayout:
        """创建左侧控制区域。"""
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(10, 0, 10, 10)
        control_layout.setSpacing(12)

        lbl_control = QLabel("控制区")
        lbl_control.setFont(QFont("Microsoft YaHei", 10))
        control_layout.addWidget(lbl_control)

        self.btn_load_model = QPushButton("1. 加载检测模型 (best.pt)")
        self.btn_load_img = QPushButton("2. 选择图片")
        self.btn_load_vid = QPushButton("3. 选择视频")
        self.btn_stop_vid = QPushButton("停止视频")
        self.btn_save = QPushButton("保存当前结果")

        buttons = [
            self.btn_load_model,
            self.btn_load_img,
            self.btn_load_vid,
            self.btn_stop_vid,
            self.btn_save,
        ]
        for button in buttons:
            button.setMinimumHeight(35)
            button.setStyleSheet(
                "QPushButton {"
                "background-color: #e1e1e1;"
                "border: 1px solid #adadad;"
                "padding: 5px;"
                "}"
                "QPushButton:hover {"
                "background-color: #e5f1fb;"
                "border: 1px solid #0078d7;"
                "}"
            )
            control_layout.addWidget(button)

        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_load_img.clicked.connect(lambda: self.load_media("image"))
        self.btn_load_vid.clicked.connect(lambda: self.load_media("video"))
        self.btn_stop_vid.clicked.connect(self.stop_video)
        self.btn_save.clicked.connect(self.save_result)

        control_layout.addWidget(QLabel("增强方式:"))
        self.combo_enhance = QComboBox()
        self.combo_enhance.addItems(["原图", "Gamma校正", "CLAHE", "Retinex", "Zero-DCE"])
        self.combo_enhance.setMinimumHeight(30)
        self.combo_enhance.currentIndexChanged.connect(self.run_image_inference)
        control_layout.addWidget(self.combo_enhance)

        control_layout.addWidget(QLabel("检测置信度阈值:"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.00)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.25)
        self.spin_conf.setMinimumHeight(30)
        self.spin_conf.valueChanged.connect(self.run_image_inference)
        control_layout.addWidget(self.spin_conf)

        control_layout.addWidget(QLabel("运行信息:"))
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setStyleSheet("background-color: white; border: 1px solid #adadad;")
        control_layout.addWidget(self.text_log)

        self.log_info("系统启动完成。请先加载训练好的行人检测模型 best.pt。")
        control_layout.addStretch(1)
        return control_layout

    def build_display_panel(self) -> QHBoxLayout:
        """创建右侧图像显示区域。"""
        display_layout = QHBoxLayout()

        original_layout = QVBoxLayout()
        original_layout.addWidget(QLabel("原始输入"))
        self.label_orig = self.create_image_label("原图 / 视频流显示区")
        original_layout.addWidget(self.label_orig, 1)

        result_layout = QVBoxLayout()
        result_layout.addWidget(QLabel("处理与检测结果"))
        self.label_res = self.create_image_label("增强 + 行人检测结果显示区")
        result_layout.addWidget(self.label_res, 1)

        display_layout.addLayout(original_layout)
        display_layout.addLayout(result_layout)
        return display_layout

    def create_image_label(self, text: str) -> QLabel:
        """创建用于显示图像的 QLabel 控件。"""
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("background-color: black; color: white; border: 2px solid #cccccc;")
        label.setMinimumSize(self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        return label

    def log_info(self, text: str) -> None:
        """在日志区域追加运行信息。"""
        self.text_log.append(text)
        scroll_bar = self.text_log.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
        QApplication.processEvents()

    def load_model(self) -> None:
        """加载 YOLO 模型权重文件。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型权重 (.pt)",
            "weights",
            "PyTorch Models (*.pt)",
        )
        if not file_path:
            return

        self.log_info(f"正在加载模型: {os.path.basename(file_path)}")
        try:
            self.model = YOLO(file_path)
            self.log_info("模型加载成功。")
        except Exception as exc:
            self.model = None
            QMessageBox.critical(self, "模型加载失败", str(exc))
            self.log_info(f"模型加载失败: {exc}")

    def load_media(self, media_type: str) -> None:
        """加载图片或视频文件并启动检测流程。"""
        if self.model is None:
            QMessageBox.warning(self, "提示", "请先加载检测模型。")
            return

        self.stop_video()
        self.annotated_image = None

        file_filter = (
            "Images (*.png *.jpg *.jpeg *.bmp)"
            if media_type == "image"
            else "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        dialog_title = "选择测试图片" if media_type == "image" else "选择测试视频"
        file_path, _ = QFileDialog.getOpenFileName(self, dialog_title, "", file_filter)
        if not file_path:
            return

        self.media_path = file_path
        self.media_type = media_type
        self.log_info(f"已载入文件: {os.path.basename(file_path)}")

        if media_type == "image":
            self.load_image_file(file_path)
        else:
            self.load_video_file(file_path)

    def load_image_file(self, file_path: str) -> None:
        """读取图片文件并执行单张图片检测。"""
        image = self.read_image(file_path)
        if image is None:
            QMessageBox.warning(self, "图片读取失败", "无法读取所选图片文件。")
            self.log_info("图片读取失败，请检查文件路径或文件格式。")
            return

        self.cv_image = image
        self.display_image(self.cv_image, self.label_orig)
        self.run_image_inference()

    def load_video_file(self, file_path: str) -> None:
        """打开视频文件并启动逐帧处理。"""
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            QMessageBox.warning(self, "视频读取失败", "无法打开所选视频文件。")
            self.log_info("视频读取失败，请检查文件路径或视频编码格式。")
            return

        self.is_playing = True
        self.log_info("开始处理视频流。")
        self.timer.start(self.VIDEO_TIMER_INTERVAL_MS)

    def run_image_inference(self) -> None:
        """对当前图片执行增强处理和行人检测。"""
        if self.media_type != "image" or self.cv_image is None or self.model is None:
            return

        enhanced_image = self.apply_enhancement(self.cv_image)
        confidence = self.spin_conf.value()

        start_time = time.time()
        results = self.model(
            enhanced_image,
            classes=[self.PERSON_CLASS_ID],
            conf=confidence,
            verbose=False,
        )
        elapsed_ms = (time.time() - start_time) * 1000

        self.annotated_image = results[0].plot()
        self.display_image(self.annotated_image, self.label_res)

        person_count = len(results[0].boxes)
        self.log_info(f"检测完成: 发现 {person_count} 个行人，耗时 {elapsed_ms:.1f} ms。")

    def process_next_frame(self) -> None:
        """读取并处理视频中的下一帧。"""
        if not self.is_playing or self.cap is None or self.model is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            self.log_info("视频播放结束。")
            return

        self.display_image(frame, self.label_orig)

        enhanced_frame = self.apply_enhancement(frame)
        results = self.model(
            enhanced_frame,
            classes=[self.PERSON_CLASS_ID],
            conf=self.spin_conf.value(),
            verbose=False,
        )
        self.annotated_image = results[0].plot()
        self.display_image(self.annotated_image, self.label_res)

    def stop_video(self) -> None:
        """停止视频播放并释放视频资源。"""
        self.is_playing = False
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def apply_enhancement(self, image: np.ndarray) -> np.ndarray:
        """根据用户选择的算法对图像进行增强。"""
        mode = self.combo_enhance.currentText()
        result = image.copy()

        if mode == "原图":
            return result
        if mode == "Gamma校正":
            return self.apply_gamma_correction(result, gamma=1.8)
        if mode == "CLAHE":
            return self.apply_clahe(result)
        if mode == "Retinex":
            return self.apply_retinex(result)
        if mode == "Zero-DCE":
            return self.apply_zero_dce_approximation(result)

        return result

    @staticmethod
    def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
        """使用 Gamma 变换提升低照度区域亮度。"""
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(image, table)

    @staticmethod
    def apply_clahe(image: np.ndarray) -> np.ndarray:
        """在 LAB 颜色空间中使用 CLAHE 增强亮度通道。"""
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        light_channel, a_channel, b_channel = cv2.split(lab_image)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_light = clahe.apply(light_channel)

        merged = cv2.merge((enhanced_light, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def apply_retinex(image: np.ndarray) -> np.ndarray:
        """使用单尺度 Retinex 方法增强图像细节。"""
        image_float = np.float64(image) + 1.0
        blur = cv2.GaussianBlur(image_float, (0, 0), 15)
        retinex = np.log10(image_float) - np.log10(blur)
        return cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def apply_zero_dce_approximation(image: np.ndarray) -> np.ndarray:
        """使用双边滤波和 Gamma 变换近似模拟 Zero-DCE 增强效果。"""
        smoothed = cv2.bilateralFilter(image, 9, 75, 75)
        inv_gamma = 1.0 / 1.5
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(smoothed, table)

    def save_result(self) -> None:
        """保存当前检测可视化结果。"""
        if self.annotated_image is None:
            QMessageBox.warning(self, "提示", "当前没有可保存的检测结果。")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存处理结果",
            "result.jpg",
            "JPEG Files (*.jpg);;PNG Files (*.png)",
        )
        if not file_path:
            return

        if self.write_image(file_path, self.annotated_image):
            self.log_info(f"结果已保存至: {file_path}")
        else:
            QMessageBox.warning(self, "保存失败", "检测结果保存失败，请检查保存路径。")
            self.log_info("检测结果保存失败。")

    @staticmethod
    def read_image(file_path: str) -> Optional[np.ndarray]:
        """读取图片，兼容包含中文字符的 Windows 文件路径。"""
        image_data = np.fromfile(file_path, dtype=np.uint8)
        if image_data.size == 0:
            return None
        return cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    @staticmethod
    def write_image(file_path: str, image: np.ndarray) -> bool:
        """保存图片，兼容包含中文字符的 Windows 文件路径。"""
        extension = os.path.splitext(file_path)[1] or ".jpg"
        success, encoded_image = cv2.imencode(extension, image)
        if not success:
            return False
        encoded_image.tofile(file_path)
        return True

    @staticmethod
    def display_image(image_bgr: np.ndarray, label_widget: QLabel) -> None:
        """将 OpenCV 的 BGR 图像转换为 Qt 图像并显示。"""
        if image_bgr is None:
            return

        if len(image_bgr.shape) == 2:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        image_rgb = np.ascontiguousarray(image_rgb)
        height, width, channels = image_rgb.shape
        bytes_per_line = channels * width

        q_image = QImage(
            image_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(q_image)
        label_widget.setPixmap(
            pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def closeEvent(self, event) -> None:
        """关闭窗口时释放视频资源。"""
        self.stop_video()
        event.accept()


# 保留旧类名，避免其他脚本引用 main_ui.AcademicDetectionUI 时失效。
AcademicDetectionUI = NightPedestrianDetectionUI


def main() -> None:
    """程序入口函数。"""
    app = QApplication(sys.argv)
    window = NightPedestrianDetectionUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
