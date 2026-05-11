import time
import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs


# =========================
# 可调参数
# =========================

WIDTH = 640
HEIGHT = 480
FPS = 30

# 高出底面多少才认为是物体，单位 m
HEIGHT_THRESHOLD = 0.008   # 8 mm

# 最高物体高度限制，防止误检，单位 m
MAX_OBJECT_HEIGHT = 0.30   # 30 cm

# 体积积分网格尺寸，单位 m
GRID_SIZE = 0.005          # 5 mm

# 是否只计算画面中间区域，None 表示使用整幅图像
# 格式: (x1, y1, x2, y2)
# 如果画面边缘有干扰，可以改成例如 (120, 80, 520, 420)
ROI = None


def start_camera():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)

    # 深度和彩色图对齐
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    return pipeline, align, depth_scale


def capture_average_frame(pipeline, align, depth_scale, frame_count=30):
    """
    连续采集多帧深度图，取中位数，降低噪声。
    """
    depth_list = []
    color_image = None
    intr = None

    # 先跳过几帧，让相机自动曝光稳定
    for _ in range(15):
        pipeline.wait_for_frames()

    for _ in range(frame_count):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
        color_image = np.asanyarray(color_frame.get_data())

        intr = depth_frame.profile.as_video_stream_profile().intrinsics

        depth_list.append(depth)

    if len(depth_list) == 0:
        raise RuntimeError("没有采集到有效深度帧，请检查 D435i 是否连接正常。")

    depth_stack = np.stack(depth_list, axis=0)
    depth_median = np.median(depth_stack, axis=0)

    return depth_median, color_image, intr


def depth_to_pointcloud(depth_image, color_image, intr, roi=None):
    """
    将深度图转成点云。
    返回:
        points: N x 3, 单位 m
        colors: N x 3, RGB, 范围 0~1
        pixels: N x 2, 对应像素坐标
    """
    h, w = depth_image.shape

    if roi is None:
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        x1, y1, x2, y2 = roi

    ys, xs = np.mgrid[y1:y2, x1:x2]

    z = depth_image[y1:y2, x1:x2]
    valid = z > 0

    xs_valid = xs[valid]
    ys_valid = ys[valid]
    z_valid = z[valid]

    x = (xs_valid - intr.ppx) / intr.fx * z_valid
    y = (ys_valid - intr.ppy) / intr.fy * z_valid

    points = np.stack([x, y, z_valid], axis=1)

    # OpenCV 是 BGR，这里转成 RGB
    bgr = color_image[ys_valid, xs_valid].astype(np.float32) / 255.0
    rgb = bgr[:, ::-1]

    pixels = np.stack([xs_valid, ys_valid], axis=1)

    return points, rgb, pixels


def make_o3d_cloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def fit_plane_by_ransac(points):
    """
    使用 RANSAC 拟合平面:
        ax + by + cz + d = 0
    """
    pcd = make_o3d_cloud(points)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.005,
        ransac_n=3,
        num_iterations=1000
    )

    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    d = d / np.linalg.norm(np.array([a, b, c], dtype=np.float64))

    return normal, d, inliers


def extract_object_points(scene_points, scene_colors, plane_normal, plane_d):
    """
    从当前点云中提取高出底面的物体点。
    """
    n = plane_normal / np.linalg.norm(plane_normal)

    # 点到平面的有符号距离
    signed_dist = scene_points @ n + plane_d

    # 不确定法向量方向，所以先取绝对距离用于验证
    height = np.abs(signed_dist)

    mask = (height > HEIGHT_THRESHOLD) & (height < MAX_OBJECT_HEIGHT)

    candidate_points = scene_points[mask]
    candidate_colors = scene_colors[mask]
    candidate_heights = height[mask]

    if len(candidate_points) == 0:
        raise RuntimeError("没有检测到高出底面的物体点，请降低 HEIGHT_THRESHOLD 或检查物体是否在画面中。")

    # 用 DBSCAN 去掉孤立噪点，只保留最大的物体簇
    candidate_pcd = make_o3d_cloud(candidate_points, candidate_colors)
    labels = np.array(candidate_pcd.cluster_dbscan(
        eps=0.025,
        min_points=30,
        print_progress=False
    ))

    valid_labels = labels[labels >= 0]

    if len(valid_labels) > 0:
        unique, counts = np.unique(valid_labels, return_counts=True)
        largest_label = unique[np.argmax(counts)]
        keep = labels == largest_label

        object_points = candidate_points[keep]
        object_colors = candidate_colors[keep]
        object_heights = candidate_heights[keep]
    else:
        # 如果聚类失败，就直接使用候选点
        object_points = candidate_points
        object_colors = candidate_colors
        object_heights = candidate_heights

    return object_points, object_colors, object_heights


def calculate_volume_height_map(object_points, object_heights, plane_normal, grid_size=0.005):
    """
    高度积分法计算体积。
    体积 ≈ Σ 每个网格的最大高度 × 网格面积
    """
    n = plane_normal / np.linalg.norm(plane_normal)

    # 构造底面平面上的两个正交方向 e1, e2
    temp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(temp, n)) > 0.9:
        temp = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(n, temp)
    e1 = e1 / np.linalg.norm(e1)

    e2 = np.cross(n, e1)
    e2 = e2 / np.linalg.norm(e2)

    # 将物体点投影到底面二维坐标
    u = object_points @ e1
    v = object_points @ e2

    u_min = u.min()
    v_min = v.min()

    ui = ((u - u_min) / grid_size).astype(np.int32)
    vi = ((v - v_min) / grid_size).astype(np.int32)

    nx = ui.max() + 1
    ny = vi.max() + 1

    flat_index = vi * nx + ui

    height_grid = np.zeros(nx * ny, dtype=np.float64)

    # 同一个网格中取最大高度，近似表示该网格的物体上表面高度
    np.maximum.at(height_grid, flat_index, object_heights)

    volume_m3 = height_grid.sum() * grid_size * grid_size

    return volume_m3


def main():
    pipeline, align, depth_scale = start_camera()

    try:
        print("\n========== 第一步：采集空底面 ==========")
        print("请先确保桌面/传送带上没有物体。")
        input("准备好后按 Enter 采集空底面...")

        empty_depth, empty_color, intr = capture_average_frame(
            pipeline, align, depth_scale, frame_count=40
        )

        empty_points, empty_colors, _ = depth_to_pointcloud(
            empty_depth, empty_color, intr, roi=ROI
        )

        print("正在拟合底面平面...")
        plane_normal, plane_d, inliers = fit_plane_by_ransac(empty_points)

        print("底面平面方程:")
        print(f"{plane_normal[0]:.6f} x + {plane_normal[1]:.6f} y + {plane_normal[2]:.6f} z + {plane_d:.6f} = 0")

        empty_pcd = make_o3d_cloud(empty_points, empty_colors)
        o3d.io.write_point_cloud("empty_plane_cloud.ply", empty_pcd)
        print("已保存空底面点云: empty_plane_cloud.ply")

        print("\n========== 第二步：采集物体 ==========")
        print("请把单个煤块/石块/不规则物体放到底面上。")
        input("放好后按 Enter 采集物体点云并计算体积...")

        object_depth, object_color, intr = capture_average_frame(
            pipeline, align, depth_scale, frame_count=40
        )

        scene_points, scene_colors, _ = depth_to_pointcloud(
            object_depth, object_color, intr, roi=ROI
        )

        scene_pcd = make_o3d_cloud(scene_points, scene_colors)
        o3d.io.write_point_cloud("scene_with_object_cloud.ply", scene_pcd)
        print("已保存完整场景点云: scene_with_object_cloud.ply")

        print("正在提取物体点云...")
        obj_points, obj_colors, obj_heights = extract_object_points(
            scene_points, scene_colors, plane_normal, plane_d
        )

        obj_pcd = make_o3d_cloud(obj_points, obj_colors)
        o3d.io.write_point_cloud("object_only_cloud.ply", obj_pcd)
        print("已保存物体点云: object_only_cloud.ply")

        print("正在计算体积...")
        volume_m3 = calculate_volume_height_map(
            obj_points,
            obj_heights,
            plane_normal,
            grid_size=GRID_SIZE
        )

        volume_cm3 = volume_m3 * 1e6
        volume_liter = volume_m3 * 1000

        print("\n========== 计算结果 ==========")
        print(f"物体点数: {len(obj_points)}")
        print(f"估算体积: {volume_m3:.8f} m^3")
        print(f"估算体积: {volume_cm3:.2f} cm^3")
        print(f"估算体积: {volume_liter:.4f} L")

        print("\n正在显示物体点云，关闭窗口后程序结束。")
        o3d.visualization.draw_geometries([obj_pcd])

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
