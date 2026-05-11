import open3d as o3d

# 读取 PLY 点云文件
pcd = o3d.io.read_point_cloud("empty_plane_cloud.ply")

# 打印点云基本信息
print(pcd)
print("点数量：", len(pcd.points))

# 可视化点云
o3d.visualization.draw_geometries([pcd])