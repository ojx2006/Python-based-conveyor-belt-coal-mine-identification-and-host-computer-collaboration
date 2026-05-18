from __future__ import annotations

"""
视觉端在线入口。

模块作用：
1. 视频输入接口：通过 OpenCV VideoCapture 读取 --source，支持本机摄像头编号
   （如 0）、视频文件路径（如 input.mp4）和 RTSP/HTTP 视频流 URL。
2. YOLO 推理模块：加载 runs/coal_v1/weights/best.pt，输出 coal/gangue 的 bbox、
   class 和 confidence。
3. 深度/体积模块：低置信度目标触发 D435i ROI 点云生成，并用高度积分估算体积。
   体积结果只写日志；tech-document.md v1.1 未定义 volume 字段，不能发给上位机。
4. 通信模块：视觉端作为 TCP Server，按 UTF-8 NDJSON 每行发送一帧 VisionResult。
"""

import argparse
import json
import logging
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "1.1"

# coal-v1 训练结果：README 与 runs/coal_v1/args.yaml 中记录为 YOLOv8s，
# 最佳权重位于 runs/coal_v1/weights/best.pt。
MODEL_WEIGHTS = ROOT / "runs" / "coal_v1" / "weights" / "best.pt"
TRAIN_ARGS_PATH = ROOT / "runs" / "coal_v1" / "args.yaml"
DEFAULT_CALIBRATION_PATH = ROOT / "config" / "calibration.yaml"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9001
DEFAULT_BELT_SPEED_MM_S = 180.0
DEFAULT_DETECT_CONF = 0.25
DEFAULT_LOW_CONF_THRESHOLD = 0.70
DEFAULT_SHARPNESS_SCORE = 0.80

# 虚拟标定参数：现场 camera_to_belt 外参未标定前，仅用于联调占位。
# TODO: 标定完成后改为读取 config/calibration.yaml 中的 R/t。
VIRTUAL_PIXEL_TO_BELT_MM = 1.0
VIRTUAL_CAMERA_TO_BELT_R = np.eye(3, dtype=np.float64)
VIRTUAL_CAMERA_TO_BELT_T_MM = np.zeros(3, dtype=np.float64)

# 虚拟底面参数：D435i 空皮带/底面未标定前，用 z=1m 的假想平面做体积积分占位。
# TODO: 现场启动时采集空皮带点云，用 RANSAC 拟合真实 plane_normal/plane_d。
VIRTUAL_PLANE_NORMAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)
VIRTUAL_PLANE_D = -1.0


@dataclass(frozen=True)
class TrainingDefaults:
    """从训练参数中复用到在线推理的少量默认值。"""

    # YOLO 输入尺寸，优先读取 runs/coal_v1/args.yaml；读取失败时用 640。
    imgsz: int = 640
    # NMS 的 IoU 阈值，和训练记录保持一致，减少线上线下差异。
    iou: float = 0.7


@dataclass
class DepthSnapshot:
    """一次对齐后的 D435i 深度快照。

    depth_m: 深度图，单位 m。
    color_bgr: 与深度图对齐后的彩色图，OpenCV BGR 顺序。
    intrinsics: RealSense 提供的相机内参，用于像素反投影到相机坐标。
    """

    depth_m: np.ndarray
    color_bgr: np.ndarray
    intrinsics: Any


@dataclass
class VolumeEstimate:
    """低置信度目标的体积估算结果。

    协议 v1.1 不允许发送 volume 字段，所以该结构只用于本地日志和后续二次判定扩展。
    """

    valid: bool
    volume_m3: float | None = None
    point_count: int = 0
    reason: str = ""


class VisionTcpServer:
    """视觉端到上位机的 TCP + NDJSON 单客户端发送模块。"""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        # _server 负责监听；_client 只保存当前连接的上位机。
        # 协议要求串行写 socket，因此这里不做多客户端广播。
        self._server: socket.socket | None = None
        self._client: socket.socket | None = None
        self._client_addr: tuple[str, int] | None = None

    def start(self) -> None:
        # SO_REUSEADDR 方便程序重启后立即重新绑定 9001 端口。
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(1)
        # 主循环要持续推理，accept 不能长时间阻塞；没有客户端时本帧只写本地日志/文件。
        self._server.settimeout(0.001)
        logging.info("vision TCP server listening on %s:%s", self.host, self.port)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._server is not None:
            self._server.close()
            self._server = None

    def _accept_if_needed(self) -> None:
        if self._server is None or self._client is not None:
            return
        try:
            # 上位机按 tech-document.md 作为 Client 主动连接视觉端。
            self._client, self._client_addr = self._server.accept()
            self._client.settimeout(1.0)
            logging.info("upper computer connected from %s", self._client_addr)
        except socket.timeout:
            return
        except OSError as exc:
            logging.warning("accept upper computer failed: %s", exc)

    def send_json(self, payload: dict[str, Any]) -> None:
        self._accept_if_needed()
        if self._client is None:
            return

        # NDJSON：每行一个完整 JSON；上位机以 \n 作为帧边界。
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            self._client.sendall(line.encode("utf-8"))
        except OSError as exc:
            # 断线不退出主程序。上位机会自动重连，下一帧重新 accept。
            logging.warning("upper computer disconnected: %s", exc)
            self._client.close()
            self._client = None
            self._client_addr = None


class D435iDepthVolumeEstimator:
    """D435i 深度采集、点云生成、皮带平面拟合和体积积分模块。"""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        volume_grid_size_m: float = 0.005,
        height_threshold_m: float = 0.008,
        max_object_height_m: float = 0.30,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        # 体积积分网格边长，单位 m；越小越细，但噪声和计算量都会增加。
        self.volume_grid_size_m = volume_grid_size_m
        # 高出底面超过该值才认为属于物体点，过滤皮带/桌面噪声。
        self.height_threshold_m = height_threshold_m
        # 过滤离底面过高的异常点，避免深度噪声把体积拉大。
        self.max_object_height_m = max_object_height_m
        # 未采集空皮带前先使用虚拟平面；--calibrate-empty-plane 会替换为实测平面。
        self.plane_normal = VIRTUAL_PLANE_NORMAL.copy()
        self.plane_d = VIRTUAL_PLANE_D
        self._rs: Any = None
        self._pipeline: Any = None
        self._align: Any = None
        self._depth_scale = 1.0
        self.enabled = False

    def start(self) -> bool:
        try:
            import pyrealsense2 as rs
        except ImportError:
            logging.warning("pyrealsense2 not installed; depth and volume use virtual fallback")
            return False

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        # 深度流和彩色流都打开，后续通过 rs.align 对齐到彩色坐标。
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        try:
            profile = self._pipeline.start(config)
        except RuntimeError as exc:
            logging.warning("D435i not available; depth and volume use virtual fallback: %s", exc)
            self._pipeline = None
            return False

        self._align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        # RealSense 原始深度是整数，需要乘 depth_scale 才是米。
        self._depth_scale = depth_sensor.get_depth_scale()
        self.enabled = True
        logging.info("D435i depth camera started")
        return True

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self.enabled = False

    def calibrate_empty_plane(self, frame_count: int = 30) -> None:
        # 启动时在空皮带/空桌面上采集一组深度帧，拟合底面平面。
        # 现场没有完成该步骤时，体积估算会退回到虚拟底面参数。
        snapshot = self.capture_average_snapshot(frame_count=frame_count)
        if snapshot is None:
            logging.warning("skip plane calibration because no valid depth snapshot was captured")
            return
        points, _, _ = self.depth_to_pointcloud(snapshot, roi=None)
        if len(points) < 100:
            logging.warning("skip plane calibration because point cloud is too sparse")
            return
        normal, plane_d = self.fit_plane_svd(points)
        self.plane_normal = normal
        self.plane_d = plane_d
        logging.info(
            "empty belt plane calibrated: %.6fx + %.6fy + %.6fz + %.6f = 0",
            normal[0],
            normal[1],
            normal[2],
            plane_d,
        )

    def capture_average_snapshot(self, frame_count: int = 1) -> DepthSnapshot | None:
        if not self.enabled or self._pipeline is None or self._align is None:
            return None

        depth_frames: list[np.ndarray] = []
        color_bgr: np.ndarray | None = None
        intrinsics: Any = None

        for _ in range(max(1, frame_count)):
            try:
                # timeout 防止相机断开时主循环永久卡住。
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue
            aligned_frames = self._align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self._depth_scale
            color_bgr = np.asanyarray(color_frame.get_data())
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            depth_frames.append(depth_m)

        if not depth_frames or color_bgr is None or intrinsics is None:
            return None

        # 多帧取中位数，比均值更能抑制偶发深度跳点。
        depth_median = np.median(np.stack(depth_frames, axis=0), axis=0)
        return DepthSnapshot(depth_m=depth_median, color_bgr=color_bgr, intrinsics=intrinsics)

    def center_to_belt_mm(self, snapshot: DepthSnapshot | None, center_pixel: tuple[int, int]) -> tuple[list[float], bool]:
        # 对 bbox 中心点取深度，反投影到相机坐标，再通过外参变换到皮带坐标。
        # 深度不可用时返回 depth_valid=False，上位机可降级为皮带平面抓取。
        if snapshot is None:
            return [0.0, 0.0, 0.0], False

        px, py = center_pixel
        if py < 0 or py >= snapshot.depth_m.shape[0] or px < 0 or px >= snapshot.depth_m.shape[1]:
            return [0.0, 0.0, 0.0], False

        depth_m = float(snapshot.depth_m[py, px])
        if depth_m <= 0.0:
            return [0.0, 0.0, 0.0], False

        camera_point_m = self.pixel_to_camera_m(px, py, depth_m, snapshot.intrinsics)
        belt_point_mm = camera_point_to_belt_mm(camera_point_m)
        # z 的语义是目标顶面相对皮带平面的高度，用局部邻域最近点估算。
        height_mm = self.estimate_top_height_mm(snapshot, center_pixel)
        if height_mm is not None:
            belt_point_mm[2] = height_mm
        return [round(float(v), 1) for v in belt_point_mm], True

    def estimate_top_height_mm(self, snapshot: DepthSnapshot, center_pixel: tuple[int, int], radius_px: int = 5) -> float | None:
        px, py = center_pixel
        # 在中心点附近取一个小窗口，使用最近深度作为目标顶面近似。
        # 这样比单点深度稍微稳一点，但现场仍需结合标定和滤波验证。
        x1 = max(0, px - radius_px)
        y1 = max(0, py - radius_px)
        x2 = min(snapshot.depth_m.shape[1], px + radius_px + 1)
        y2 = min(snapshot.depth_m.shape[0], py + radius_px + 1)
        roi_depth = snapshot.depth_m[y1:y2, x1:x2]
        valid = roi_depth[roi_depth > 0]
        if len(valid) == 0:
            return None

        nearest_depth_m = float(valid.min())
        point_m = self.pixel_to_camera_m(px, py, nearest_depth_m, snapshot.intrinsics)
        signed_dist_m = float(point_m @ self.plane_normal + self.plane_d)
        return max(0.0, abs(signed_dist_m) * 1000.0)

    @staticmethod
    def pixel_to_camera_m(px: int, py: int, depth_m: float, intrinsics: Any) -> np.ndarray:
        # 针孔相机反投影：
        # X=(u-cx)/fx*Z, Y=(v-cy)/fy*Z, Z=depth。
        x_m = (px - intrinsics.ppx) / intrinsics.fx * depth_m
        y_m = (py - intrinsics.ppy) / intrinsics.fy * depth_m
        return np.array([x_m, y_m, depth_m], dtype=np.float64)

    def depth_to_pointcloud(
        self,
        snapshot: DepthSnapshot,
        roi: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 将 ROI 内每个有效深度像素转换成 3D 点。
        # 返回 pixels 是为了后续如需把点云结果映射回图像时可直接使用。
        depth_image = snapshot.depth_m
        color_image = snapshot.color_bgr
        h, w = depth_image.shape

        if roi is None:
            x1, y1, x2, y2 = 0, 0, w, h
        else:
            x1, y1, x2, y2 = clamp_roi(roi, w, h)

        ys, xs = np.mgrid[y1:y2, x1:x2]
        z = depth_image[y1:y2, x1:x2]
        valid = z > 0
        if not valid.any():
            return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2), dtype=np.int32)

        xs_valid = xs[valid]
        ys_valid = ys[valid]
        z_valid = z[valid]
        intr = snapshot.intrinsics

        x = (xs_valid - intr.ppx) / intr.fx * z_valid
        y = (ys_valid - intr.ppy) / intr.fy * z_valid
        points = np.stack([x, y, z_valid], axis=1)

        colors = color_image[ys_valid, xs_valid].astype(np.float32) / 255.0
        pixels = np.stack([xs_valid, ys_valid], axis=1)
        return points, colors, pixels

    @staticmethod
    def fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float]:
        # 简化版平面拟合：对点云做 SVD，最小奇异值方向就是平面法向。
        # 这里没有用 RANSAC，速度快但对离群点更敏感；正式现场建议换回 RANSAC。
        sample_step = max(1, len(points) // 10000)
        sampled = points[::sample_step]
        centroid = sampled.mean(axis=0)
        _, _, vh = np.linalg.svd(sampled - centroid, full_matrices=False)
        normal = vh[-1]
        normal = normal / np.linalg.norm(normal)
        plane_d = -float(normal @ centroid)
        return normal, plane_d

    def extract_object_points(self, scene_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(scene_points) == 0:
            return np.empty((0, 3)), np.empty((0,))

        # 点到平面的距离近似目标高度。当前取绝对值，不假设法向量方向。
        normal = self.plane_normal / np.linalg.norm(self.plane_normal)
        signed_dist = scene_points @ normal + self.plane_d
        heights = np.abs(signed_dist)
        mask = (heights > self.height_threshold_m) & (heights < self.max_object_height_m)
        return scene_points[mask], heights[mask]

    def calculate_volume_height_map(self, object_points: np.ndarray, object_heights: np.ndarray) -> float:
        if len(object_points) == 0:
            return 0.0

        # 在底面平面内构造二维坐标系，把物体点投影到网格上。
        # 每个网格取最大高度，最后 Σ(高度 * 网格面积) 得到体积近似。
        normal = self.plane_normal / np.linalg.norm(self.plane_normal)
        temp = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(temp, normal))) > 0.9:
            temp = np.array([0.0, 1.0, 0.0])

        e1 = np.cross(normal, temp)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        e2 = e2 / np.linalg.norm(e2)

        u = object_points @ e1
        v = object_points @ e2
        ui = ((u - u.min()) / self.volume_grid_size_m).astype(np.int32)
        vi = ((v - v.min()) / self.volume_grid_size_m).astype(np.int32)

        nx = int(ui.max()) + 1
        flat_index = vi * nx + ui
        height_grid = np.zeros(int(flat_index.max()) + 1, dtype=np.float64)
        np.maximum.at(height_grid, flat_index, object_heights)
        return float(height_grid.sum() * self.volume_grid_size_m * self.volume_grid_size_m)

    def estimate_volume_for_bbox(
        self,
        bbox_xywh: list[int],
        frame_shape: tuple[int, int],
        frame_count: int,
    ) -> VolumeEstimate:
        # 只在低置信度目标上调用，避免每个目标都做点云积分拖慢主循环。
        snapshot = self.capture_average_snapshot(frame_count=frame_count)
        if snapshot is None:
            return VolumeEstimate(valid=False, reason="no valid D435i depth snapshot")

        # YOLO 视频帧和 D435i 深度帧分辨率可能不同，需要按比例缩放 bbox 到深度图。
        roi = scale_bbox_to_depth_roi(bbox_xywh, frame_shape, snapshot.depth_m.shape)
        scene_points, _, _ = self.depth_to_pointcloud(snapshot, roi=roi)
        object_points, object_heights = self.extract_object_points(scene_points)
        if len(object_points) == 0:
            return VolumeEstimate(valid=False, reason="no object points above belt plane")

        volume_m3 = self.calculate_volume_height_map(object_points, object_heights)
        return VolumeEstimate(valid=True, volume_m3=volume_m3, point_count=len(object_points))


def camera_point_to_belt_mm(camera_point_m: np.ndarray) -> np.ndarray:
    # 相机坐标单位是 m，协议要求输出皮带坐标系 mm。
    # 当前默认 R=I、t=0，仅作为未标定前的占位；有 calibration.yaml 时会覆盖。
    camera_point_mm = camera_point_m * 1000.0
    return VIRTUAL_CAMERA_TO_BELT_R @ camera_point_mm + VIRTUAL_CAMERA_TO_BELT_T_MM


def load_camera_to_belt_calibration(path: Path) -> None:
    global VIRTUAL_CAMERA_TO_BELT_R, VIRTUAL_CAMERA_TO_BELT_T_MM

    # tech-document.md 约定由上位机交付 config/calibration.yaml。
    # 文件不存在时不阻塞联调，但所有 world_coord_mm 都只能视为虚拟坐标。
    if not path.exists():
        logging.warning("calibration file not found; using virtual camera_to_belt R/t: %s", path)
        return

    try:
        import yaml
    except ImportError:
        logging.warning("PyYAML not installed; using virtual camera_to_belt R/t")
        return

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # 期望格式：
    # vision:
    #   camera_to_belt:
    #     R: [[...], [...], [...]]
    #     t: [tx, ty, tz]
    camera_to_belt = data.get("vision", {}).get("camera_to_belt", {})
    rotation = camera_to_belt.get("R")
    translation = camera_to_belt.get("t")
    if rotation is None or translation is None:
        logging.warning("invalid calibration file; using virtual camera_to_belt R/t: %s", path)
        return

    try:
        VIRTUAL_CAMERA_TO_BELT_R = np.array(rotation, dtype=np.float64).reshape(3, 3)
        VIRTUAL_CAMERA_TO_BELT_T_MM = np.array(translation, dtype=np.float64).reshape(3)
    except ValueError as exc:
        logging.warning("invalid calibration matrix shape; using virtual camera_to_belt R/t: %s", exc)
        return
    logging.info("loaded camera_to_belt calibration: %s", path)


def load_training_defaults(path: Path) -> TrainingDefaults:
    # 不引入完整 YAML 解析依赖来读训练 args；这里只需要 imgsz/iou 两个简单键。
    # 若 args.yaml 不存在或字段缺失，就回落到 TrainingDefaults。
    defaults = TrainingDefaults()
    if not path.exists():
        return defaults

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")

    imgsz = int(float(values.get("imgsz", defaults.imgsz)))
    iou = float(values.get("iou", defaults.iou))
    return TrainingDefaults(imgsz=imgsz, iou=iou)


def parse_source(source: str) -> str | int:
    # OpenCV 对摄像头编号要求 int，对视频路径/RTSP URL 要求 str。
    if source.isdigit():
        return int(source)
    return source


def clamp_roi(roi: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    # 防止 YOLO 框或缩放后的深度 ROI 越界；同时保证 x2>x1、y2>y1。
    x1, y1, x2, y2 = roi
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def scale_bbox_to_depth_roi(
    bbox_xywh: list[int],
    video_shape: tuple[int, int],
    depth_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    # YOLO 使用视频帧坐标，D435i 使用深度帧坐标。
    # 如果两者分辨率不同，用宽高比例把 bbox 映射到深度图。
    video_h, video_w = video_shape
    depth_h, depth_w = depth_shape
    x, y, w, h = bbox_xywh
    sx = depth_w / max(1, video_w)
    sy = depth_h / max(1, video_h)
    return (
        int(x * sx),
        int(y * sy),
        int((x + w) * sx),
        int((y + h) * sy),
    )


def scale_pixel_to_depth(
    center_pixel: list[int],
    video_shape: tuple[int, int],
    depth_shape: tuple[int, int],
) -> tuple[int, int]:
    # 和 bbox 缩放同理，把视频帧中心点映射到深度图中心点。
    video_h, video_w = video_shape
    depth_h, depth_w = depth_shape
    px = int(center_pixel[0] * depth_w / max(1, video_w))
    py = int(center_pixel[1] * depth_h / max(1, video_h))
    return max(0, min(depth_w - 1, px)), max(0, min(depth_h - 1, py))


def format_timestamp() -> str:
    # 协议要求本机时间，格式 YYYY-MM-DD HH:MM:SS.fff。
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def sharpness_score(frame_bgr: np.ndarray) -> float:
    # 用 Laplacian 方差做简易清晰度评分：越大通常越清晰。
    # 这里压缩到 0~1，方便满足协议字段；正式阈值需结合现场运动速度调参。
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return round(min(1.0, variance / 1000.0), 3)
    except cv2.error:
        return DEFAULT_SHARPNESS_SCORE


def virtual_belt_coord_from_pixel(center_pixel: tuple[int, int], image_width: int, image_height: int) -> list[float]:
    cx, cy = center_pixel
    # 虚拟坐标换算：未完成外参和深度标定前，仅把像素中心粗略映射到皮带平面。
    # TODO: 用 D435i 深度反投影 + camera_to_belt 标定矩阵替换。
    x_mm = (cy - image_height / 2.0) * VIRTUAL_PIXEL_TO_BELT_MM
    y_mm = (cx - image_width / 2.0) * VIRTUAL_PIXEL_TO_BELT_MM
    z_mm = 0.0
    return [round(x_mm, 1), round(y_mm, 1), z_mm]


def make_result_frame(
    frame_id: int,
    image_width: int,
    image_height: int,
    belt_speed_mm_s: float,
    objects: list[dict[str, Any]],
) -> dict[str, Any]:
    # 顶层字段必须和 tech-document.md v1.1 完全一致。
    # 上位机 pydantic extra="forbid"，不能额外增加调试字段。
    return {
        "protocol_version": PROTOCOL_VERSION,
        "frame_id": frame_id,
        "timestamp": format_timestamp(),
        "image_width": image_width,
        "image_height": image_height,
        "belt_speed_mm_s": float(belt_speed_mm_s),
        "objects": objects,
    }


def build_objects(
    model_names: dict[int, str],
    result: Any,
    frame_bgr: np.ndarray,
    depth_estimator: D435iDepthVolumeEstimator,
    depth_snapshot: DepthSnapshot | None,
    low_conf_threshold: float,
    volume_frames: int,
) -> list[dict[str, Any]]:
    """把 YOLO 原始输出转换为 tech-document.md v1.1 允许的 objects[] 字段。"""

    image_height, image_width = frame_bgr.shape[:2]
    frame_sharpness = sharpness_score(frame_bgr)
    objects: list[dict[str, Any]] = []

    boxes = result.boxes
    if boxes is None:
        return objects

    for box in boxes:
        # Ultralytics 输出的 class_id 需要映射回训练时的类别名。
        class_id = int(box.cls[0].item())
        class_name = str(model_names.get(class_id, class_id))
        # 协议只允许 coal/gangue；训练集中 objects 类仅用于数据标注辅助，不能发给上位机。
        if class_name not in {"coal", "gangue"}:
            logging.debug("skip unsupported class for protocol v1.1: %s", class_name)
            continue

        confidence = float(box.conf[0].item())
        # xyxy 转协议要求的 bbox=[x,y,w,h]，其中 x/y 是左上角。
        x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = clamp_roi((x1, y1, x2, y2), image_width, image_height)
        bbox = [x1, y1, x2 - x1, y2 - y1]
        center_pixel = [x1 + bbox[2] // 2, y1 + bbox[3] // 2]

        depth_center_pixel = (center_pixel[0], center_pixel[1])
        if depth_snapshot is not None:
            # 深度帧分辨率可能与视频帧不同，中心点也要同步缩放。
            depth_center_pixel = scale_pixel_to_depth(
                center_pixel,
                video_shape=(image_height, image_width),
                depth_shape=depth_snapshot.depth_m.shape,
            )
        world_coord_mm, depth_valid = depth_estimator.center_to_belt_mm(depth_snapshot, depth_center_pixel)
        if not depth_valid:
            # 没有深度时仍需给 world_coord_mm 三个数；z=0 表示退化到皮带平面。
            world_coord_mm = virtual_belt_coord_from_pixel((center_pixel[0], center_pixel[1]), image_width, image_height)

        # id 仅要求当前帧内唯一；跳过 objects 类后重新连续编号。
        protocol_object_id = len(objects) + 1

        if confidence < low_conf_threshold:
            # 低置信度进入体积复核分支。由于协议未定义 volume，只记录日志。
            estimate = depth_estimator.estimate_volume_for_bbox(
                bbox,
                frame_shape=(image_height, image_width),
                frame_count=volume_frames,
            )
            if estimate.valid:
                logging.info(
                    "low confidence object id=%s class=%s conf=%.3f volume=%.8fm^3 %.2fcm^3 points=%s",
                    protocol_object_id,
                    class_name,
                    confidence,
                    estimate.volume_m3,
                    estimate.volume_m3 * 1e6 if estimate.volume_m3 is not None else 0.0,
                    estimate.point_count,
                )
            else:
                logging.info(
                    "low confidence object id=%s class=%s conf=%.3f volume unavailable: %s",
                    protocol_object_id,
                    class_name,
                    confidence,
                    estimate.reason,
                )

        objects.append(
            {
                "id": protocol_object_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": bbox,
                "center_pixel": center_pixel,
                "world_coord_mm": world_coord_mm,
                "depth_valid": depth_valid,
                "sharpness_score": frame_sharpness,
                # 这是视觉端给上位机的建议位；最终是否抓取仍由上位机状态机决定。
                "send_to_robot": confidence >= low_conf_threshold,
            }
        )

    return objects


def run(args: argparse.Namespace) -> None:
    """主循环：读视频帧、推理、按协议组包并串行发送给上位机。"""

    training_defaults = load_training_defaults(TRAIN_ARGS_PATH)
    # 命令行参数优先；未传入时复用 coal_v1 训练记录中的 imgsz/iou。
    imgsz = args.imgsz or training_defaults.imgsz
    iou = args.iou if args.iou is not None else training_defaults.iou

    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"coal-v1 weight file not found: {MODEL_WEIGHTS}")

    load_camera_to_belt_calibration(Path(args.calibration))

    model = YOLO(str(MODEL_WEIGHTS))
    model_names = {int(key): value for key, value in model.names.items()}
    logging.info("loaded YOLO coal-v1 model: %s", MODEL_WEIGHTS)
    logging.info("inference params: imgsz=%s conf=%.2f iou=%.2f", imgsz, args.conf, iou)

    depth_estimator = D435iDepthVolumeEstimator(
        width=args.depth_width,
        height=args.depth_height,
        fps=args.depth_fps,
        volume_grid_size_m=args.volume_grid_size,
    )
    if not args.no_depth:
        # 深度相机不可用时不会中断 YOLO/TCP 主链路，只把 depth_valid 置 false。
        depth_estimator.start()
        if args.calibrate_empty_plane and depth_estimator.enabled:
            depth_estimator.calibrate_empty_plane(frame_count=args.plane_frames)

    server = VisionTcpServer(args.host, args.port)
    server.start()

    # 视频输入统一从 --source 进入：
    # 摄像头编号、本地视频文件和 RTSP/HTTP 流都由 OpenCV VideoCapture 处理。
    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        server.close()
        depth_estimator.stop()
        raise RuntimeError(f"cannot open video source: {args.source}")

    frame_id = 0
    last_sent_at = 0.0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                logging.info("video source ended")
                break

            # 每帧取一次深度快照。禁用深度或相机不可用时返回 None。
            depth_snapshot = depth_estimator.capture_average_snapshot(frame_count=args.depth_frames)
            results = model.predict(
                source=frame_bgr,
                imgsz=imgsz,
                conf=args.conf,
                iou=iou,
                verbose=False,
            )
            result = results[0]
            # 将 YOLO 输出整理成协议 objects[]，包含坐标、深度有效性和抓取建议。
            objects = build_objects(
                model_names=model_names,
                result=result,
                frame_bgr=frame_bgr,
                depth_estimator=depth_estimator,
                depth_snapshot=depth_snapshot,
                low_conf_threshold=args.low_conf_threshold,
                volume_frames=args.volume_frames,
            )

            image_height, image_width = frame_bgr.shape[:2]
            payload = make_result_frame(
                frame_id=frame_id,
                image_width=image_width,
                image_height=image_height,
                belt_speed_mm_s=args.belt_speed,
                objects=objects,
            )
            # TCP 无客户端时 send_json 会静默跳过；有客户端时发送一行 NDJSON。
            server.send_json(payload)

            if args.jsonl:
                # 本地记录用于协议自测/回放，不影响 TCP 发送。
                args.jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")
                args.jsonl.flush()

            now = time.time()
            if now - last_sent_at >= 1.0:
                logging.info("frame=%s objects=%s", frame_id, len(objects))
                last_sent_at = now

            if args.show:
                # 本地预览只用于调试，现场可关闭以减少 UI 开销。
                annotated = result.plot()
                cv2.imshow("vision-coal-gangue", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1
            if args.max_frames is not None and frame_id >= args.max_frames:
                break
    except Exception:
        # 推理异常时按协议推一个空帧，避免上位机误以为收到半包 JSON。
        logging.exception("inference loop failed; send empty frame before exit")
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
        server.send_json(make_result_frame(frame_id, w, h, args.belt_speed, []))
        raise
    finally:
        # 所有硬件/窗口/socket 都在 finally 释放，避免下次启动端口或相机被占用。
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        server.close()
        depth_estimator.stop()
        if args.jsonl:
            args.jsonl.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO coal/gangue video inference and upper-computer TCP output.")
    # 视频输入接口：
    # - "0" / "1": 本机摄像头编号
    # - "D:\\path\\input.mp4": 本地视频文件
    # - "rtsp://...": 网络视频流
    parser.add_argument("--source", default="0", help="camera index, video file path, or RTSP/HTTP stream URL")
    parser.add_argument("--host", default=DEFAULT_HOST, help="TCP listen host for upper computer")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP listen port for upper computer")
    parser.add_argument("--conf", type=float, default=DEFAULT_DETECT_CONF, help="YOLO detection confidence threshold")
    parser.add_argument("--low-conf-threshold", type=float, default=DEFAULT_LOW_CONF_THRESHOLD, help="trigger volume estimation below this confidence")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference size; default reads runs/coal_v1/args.yaml")
    parser.add_argument("--iou", type=float, default=None, help="YOLO NMS IoU; default reads runs/coal_v1/args.yaml")
    parser.add_argument("--belt-speed", type=float, default=DEFAULT_BELT_SPEED_MM_S, help="belt speed in mm/s")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH), help="camera_to_belt calibration YAML")
    parser.add_argument("--jsonl", type=argparse.FileType("w", encoding="utf-8"), default=None, help="optional local NDJSON recording path")
    parser.add_argument("--show", action="store_true", help="show annotated local preview window")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N frames; useful for smoke test")
    parser.add_argument("--no-depth", action="store_true", help="disable D435i depth and volume branch")
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--depth-fps", type=int, default=30)
    parser.add_argument("--depth-frames", type=int, default=1, help="D435i frames averaged for coordinate depth")
    parser.add_argument("--volume-frames", type=int, default=8, help="D435i frames averaged for low-confidence volume")
    parser.add_argument("--volume-grid-size", type=float, default=0.005, help="volume integration grid size in meters")
    parser.add_argument("--calibrate-empty-plane", action="store_true", help="fit belt plane from an empty-belt depth snapshot at startup")
    parser.add_argument("--plane-frames", type=int, default=30, help="frames used by --calibrate-empty-plane")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
