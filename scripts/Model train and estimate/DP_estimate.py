import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import os

# ==========================================
# 核心指标计算函数
# ==========================================

def calculate_rmse(orig_geom, simp_geom):
    """计算原始节点到简化几何体的均方根误差 (RMSE)"""
    if orig_geom.geom_type == 'Polygon':
        coords = list(orig_geom.exterior.coords)
    elif orig_geom.geom_type == 'LineString':
        coords = list(orig_geom.coords)
    else:
        return np.nan

    distances = [Point(c).distance(simp_geom) for c in coords]
    return np.sqrt(np.mean(np.square(distances)))


def calculate_menger_curvature(points):
    """计算离散点的 Menger 曲率"""
    k_values = []
    n = len(points)
    for i in range(1, n - 1):
        p_prev = np.array(points[i - 1])
        p_curr = np.array(points[i])
        p_next = np.array(points[i + 1])

        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)

        if len_v1 == 0 or len_v2 == 0:
            k_values.append(0.0)
            continue

        cos_theta = np.dot(v1, v2) / (len_v1 * len_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        k = (2 * np.sin(theta)) / (len_v1 + len_v2)
        k_values.append(k)

    return np.array(k_values)


def calculate_curvature_error(geom_P, geom_Q, num_samples=500):
    """计算等弧长重采样后的平均曲率变化误差 (\Delta \kappa)"""
    # 提取外部边界用于等距离重采样
    boundary_P = geom_P.exterior if geom_P.geom_type == 'Polygon' else geom_P
    boundary_Q = geom_Q.exterior if geom_Q.geom_type == 'Polygon' else geom_Q

    distances_P = np.linspace(0, boundary_P.length, num_samples)
    resampled_P = [boundary_P.interpolate(d).coords[0] for d in distances_P]

    distances_Q = np.linspace(0, boundary_Q.length, num_samples)
    resampled_Q = [boundary_Q.interpolate(d).coords[0] for d in distances_Q]

    k_P = calculate_menger_curvature(resampled_P)
    k_Q = calculate_menger_curvature(resampled_Q)

    return np.mean(np.abs(k_P - k_Q))


# ==========================================
# 主评估流程
# ==========================================

def evaluate_compression(orig_shp_path, simp_shp_path):
    """读取两个 Shapefile 并计算整体评估指标"""
    print(f"加载原始数据: {orig_shp_path}")
    gdf_orig = gpd.read_file(orig_shp_path)

    print(f"加载压缩数据: {simp_shp_path}")
    gdf_simp = gpd.read_file(simp_shp_path)

    # 确保两者要素数量一致
    if len(gdf_orig) != len(gdf_simp):
        print("警告: 原始文件与简化文件的要素数量不一致，评估可能不准确！")

    # 初始化指标存储列表
    metrics = {
        'rmse': [], 'iou': [], 'delta_kappa': [],
        'area_orig': [], 'area_simp': [],
        'perim_orig': [], 'perim_simp': []
    }

    # 逐要素对比计算
    for idx, (geom_orig, geom_simp) in enumerate(zip(gdf_orig.geometry, gdf_simp.geometry)):
        if geom_orig is None or geom_simp is None or geom_orig.is_empty or geom_simp.is_empty:
            continue

        # 统一转换为 Polygon (计算面积极其重要)
        poly_orig = Polygon(geom_orig) if geom_orig.geom_type == 'LineString' else geom_orig
        poly_simp = Polygon(geom_simp) if geom_simp.geom_type == 'LineString' else geom_simp

        # 1. 面积与周长基础数据
        metrics['area_orig'].append(poly_orig.area)
        metrics['area_simp'].append(poly_simp.area)
        metrics['perim_orig'].append(poly_orig.length)
        metrics['perim_simp'].append(poly_simp.length)

        # 2. RMSE
        metrics['rmse'].append(calculate_rmse(geom_orig, geom_simp))

        # 3. IoU
        intersection = poly_orig.intersection(poly_simp).area
        union = poly_orig.union(poly_simp).area
        metrics['iou'].append(intersection / union if union > 0 else 0)

        # 4. 曲率误差
        metrics['delta_kappa'].append(calculate_curvature_error(poly_orig, poly_simp))

    # ==========================================
    # 汇总与平均值计算
    # ==========================================

    # 计算 EA (平均面积) 和 EP (平均周长)
    EA_orig = np.mean(metrics['area_orig'])
    EA_simp = np.mean(metrics['area_simp'])
    EP_orig = np.mean(metrics['perim_orig'])
    EP_simp = np.mean(metrics['perim_simp'])

    # 计算相对误差 (基于整体平均值，避免局部极小面积产生的除零或极值干扰)
    delta_A = abs(EA_orig - EA_simp) / EA_orig if EA_orig > 0 else 0
    delta_L = abs(EP_orig - EP_simp) / EP_orig if EP_orig > 0 else 0

    mean_rmse = np.mean(metrics['rmse'])
    mean_iou = np.mean(metrics['iou'])
    mean_kappa = np.mean(metrics['delta_kappa'])

    # 打印最终报告
    print("\n" + "=" * 40)
    print(" 实验精度评估报告 (Dataset Averages)")
    print("=" * 40)
    print(f"均方根误差 (RMSE):\t {mean_rmse:.6f}")
    print(f"交并比 (IoU):\t\t {mean_iou:.6f}")
    print(f"曲率变化误差 (Δκ):\t {mean_kappa:.6f}")
    print("-" * 40)
    print(f"原始平均面积 (EA_orig):\t {EA_orig:.2f}")
    print(f"简化平均面积 (EA_simp):\t {EA_simp:.2f}")
    print(f"相对面积误差 (δA):\t {delta_A * 100:.4f}%")
    print("-" * 40)
    print(f"原始平均周长 (EP_orig):\t {EP_orig:.2f}")
    print(f"简化平均周长 (EP_simp):\t {EP_simp:.2f}")
    print(f"相对周长误差 (δL):\t {delta_L * 100:.4f}%")
    print("=" * 40)


# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    # 替换为你的实际路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', '..', 'data')

    # 替换为你实际的文件路径
    original_shapefile = os.path.join(data_dir, "Line_Island_part_DP.shp")
    simplified_shapefile = os.path.join(data_dir, "DP_test_220.shp")

    evaluate_compression(original_shapefile, simplified_shapefile)