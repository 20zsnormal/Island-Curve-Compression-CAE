import math
import torch
from torch import nn
import os
import numpy as np
import time
from shapely.geometry import Polygon  # 新增：用于计算多边形面积、周长和IoU
import pandas as pd  # 新增：用于导出数据表

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Selected device: {device}')


# ==========================================
# 1. 模型定义
# ==========================================
# class Encode_Decode(nn.Module):
#     def __init__(self, encoded_space_dim=100):
#         super(Encode_Decode, self).__init__()
#         # Encoder
#         self.Encode = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=5, padding=1),
#             nn.BatchNorm1d(16),
#             nn.ReLU(True),
#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=5, padding=1),
#             nn.BatchNorm1d(32),
#             nn.ReLU(True),
#         )
#         self.flatten = nn.Flatten(start_dim=1)
#         self.encoder_lin = nn.Sequential(
#             nn.Linear(32 * 8, encoded_space_dim),
#             nn.ReLU(True),
#         )
#
#         # Decoder
#         self.decoder_lin = nn.Sequential(
#             nn.Linear(encoded_space_dim, 32 * 8),
#             nn.ReLU(True),
#         )
#         self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 8))
#         self.Decode = nn.Sequential(
#             nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=7, stride=5, padding=1, output_padding=0),
#             nn.BatchNorm1d(16),
#             nn.ReLU(True),
#             nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=7, stride=5, padding=1, output_padding=0),
#         )
#
#     def forward(self, x):
#         x = self.Encode(x)
#         x = self.flatten(x)
#         compress = self.encoder_lin(x)
#         x = self.decoder_lin(compress)
#         x = self.unflatten(x)
#         x = self.Decode(x)
#         return x
# class Encode_Decode(nn.Module):
#
#     def __init__(self,encoded_space_dim=60):
#         super(Encode_Decode, self).__init__()
#         # Encoder
#         self.Encode = nn.Sequential(
#             nn.Linear(200, 150),
#             nn.BatchNorm1d(1),  # 添加批量归一化层
#             nn.ReLU(True),
#             nn.Linear(150, 150),
#             nn.BatchNorm1d(1),  # 添加批量归一化层
#             nn.ReLU(True),
#
#         )
#         ### Linear p
#         self.encoder_lin = nn.Sequential(
#             nn.Linear( 150, encoded_space_dim),
#             nn.BatchNorm1d(1),  # 添加批量归一化层
#             nn.ReLU(True),
#         )
#
#         # Decoder
#         self.decoder_lin = nn.Sequential(
#             nn.Linear(encoded_space_dim, 150),
#             nn.ReLU(True),
#
#         )
#         self.Decode = nn.Sequential(
#             nn.Linear(150, 150),
#             nn.BatchNorm1d(1),  # 添加批量归一化层
#             nn.ReLU(True),
#             nn.Linear(150, 200),
#         )
#     def forward(self,x):
#         x=self.Encode(x)
#         compress=self.encoder_lin(x)
#         x=self.decoder_lin(compress)
#         x=self.Decode(x)
#         return x

# ----------------------------------------------------------------------
# Helper Function: Create Normalized Adjacency Matrix for a Line Graph
# ----------------------------------------------------------------------
def create_normalized_adj(num_nodes=100):
    """
    生成一个具有100个节点的线形图的归一化邻接矩阵。
    包括自环 (Self-loops)，以保留节点自身的特征。
    """
    # 1. 创建基础邻接矩阵 A (无向图，相邻节点连通)
    A = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0

    # 2. 增加自环 (A_tilde = A + I)
    A_tilde = A + np.eye(num_nodes)

    # 3. 计算度矩阵 D
    D = np.sum(A_tilde, axis=1)

    # 4. 计算 D^{-1/2}
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_mat_inv_sqrt = np.diag(D_inv_sqrt)

    # 5. 计算归一化邻接矩阵: D^{-1/2} * A_tilde * D^{-1/2}
    norm_adj = D_mat_inv_sqrt.dot(A_tilde).dot(D_mat_inv_sqrt)

    return torch.tensor(norm_adj, dtype=torch.float32)


# ----------------------------------------------------------------------
# Standard Graph Convolutional Layer
# ----------------------------------------------------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x 形状: [Batch, num_nodes, in_features]
        # adj 形状: [num_nodes, num_nodes]
        # 1. 特征变换
        support = self.linear(x)
        # 2. 邻居信息聚合 (PyTorch 会自动对 Batch 维度进行广播)
        out = torch.matmul(adj, support)
        return out


# ----------------------------------------------------------------------
# Graph Convolutional Autoencoder
# ----------------------------------------------------------------------

class Encode_Decode(nn.Module):
    """
    基于图卷积 (GCN) 的自编码器。
    作为 1D-CNN 的无缝替换，内部自动进行数据维度的重排。
    """

    def __init__(self, num_nodes=100, in_features=2, encoded_space_dim=60):
        super(Encode_Decode, self).__init__()

        self.num_nodes = num_nodes
        self.in_features = in_features

        # 将邻接矩阵注册为 buffer，这样它会自动随模型移动到 GPU，并包含在 state_dict 中
        self.register_buffer('adj', create_normalized_adj(num_nodes))

        # ---------------- Encoder ----------------
        self.gcn1 = GCNLayer(in_features, 16)
        # 修复：BatchNorm 的参数应该是特征维度 (16)，而不是节点数
        self.bn1 = nn.BatchNorm1d(16)

        self.gcn2 = GCNLayer(16, 32)
        # 修复：BatchNorm 的参数应该是特征维度 (32)
        self.bn2 = nn.BatchNorm1d(32)

        # Flatten 后的全连接层
        self.encoder_lin = nn.Sequential(
            nn.Linear(num_nodes * 32, encoded_space_dim),
            nn.ReLU(True),
        )

        # ---------------- Decoder ----------------
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, num_nodes * 32),
            nn.ReLU(True),
        )

        self.gcn3 = GCNLayer(32, 16)
        # 修复：BatchNorm 的参数应该是特征维度 (16)
        self.bn3 = nn.BatchNorm1d(16)

        self.gcn4 = GCNLayer(16, in_features)

    def forward(self, x):
        """
        x: [Batch, 1, 200] (继承你原有的输入格式)
        """
        batch_size = x.size(0)

        # 1. 数据重排：转换为图节点格式 [Batch, 100, 2]
        h = x.view(batch_size, self.num_nodes, self.in_features)

        # --- 编码阶段 ---
        h = self.gcn1(h, self.adj)
        # BatchNorm1d 期望的输入形状是 [Batch, Features, Nodes]，所以使用 transpose
        h = torch.relu(self.bn1(h.transpose(1, 2)).transpose(1, 2))

        h = self.gcn2(h, self.adj)
        h = torch.relu(self.bn2(h.transpose(1, 2)).transpose(1, 2))

        # 压平并映射到潜空间
        h_flat = h.contiguous().view(batch_size, -1)
        compress = self.encoder_lin(h_flat)

        # --- 解码阶段 ---
        h_dec = self.decoder_lin(compress)
        h_dec = h_dec.view(batch_size, self.num_nodes, 32)

        h_dec = self.gcn3(h_dec, self.adj)
        h_dec = torch.relu(self.bn3(h_dec.transpose(1, 2)).transpose(1, 2))

        out = self.gcn4(h_dec, self.adj)  # 输出形状: [Batch, 100, 2]

        # 2. 恢复原有形状：转换为 [Batch, 1, 200] 适配原有的损失计算
        out_flat = out.view(batch_size, 1, -1)

        return out_flat
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(f'Selected device: {device}')

# ==========================================
# 2. 数据处理与辅助函数
# ==========================================
def load_txt_files(folder_path):
    file_list = os.listdir(folder_path)
    sample_num = len(file_list)
    all_data = []
    for i in range(int(sample_num)):
        file_name = "diff_" + str(i + 1) + "_resampled_coordinates.txt"
        file_path = os.path.join(folder_path, file_name)
        data = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                for line in file:
                    values = list(map(float, line.strip().split(',')))
                    data.extend(values)
            if len(data) == 200:
                reshaped_data = np.array(data).reshape(200, 1)
                all_data.append(reshaped_data)
    all_data_array = np.stack(all_data)
    return all_data_array

# 获取起点坐标
def get_start_point(path):
    file_list = [f for f in os.listdir(path) if f.endswith("_resampled_coordinates.txt")]
    sample_num = len(file_list)

    samples = []
    for i in range(sample_num):
        txt_path = os.path.join(path, f"{i + 1}_resampled_coordinates.txt")
        with open(txt_path, "r") as f:
            ori_data = f.read().strip().split("\n")
            samples.append(ori_data[0])  # 获取第一行
    return np.array(samples)

# 获取终点坐标
def get_last_point(path):
    file_list = [f for f in os.listdir(path) if f.endswith("_resampled_coordinates.txt")]
    sample_num = len(file_list)

    samples = []
    for i in range(sample_num):
        txt_path = os.path.join(path, f"{i + 1}_resampled_coordinates.txt")
        with open(txt_path, "r") as f:
            ori_data = f.read().strip().split("\n")
            samples.append(ori_data[-1])  # 获取最后一行的坐标
    return np.array(samples)

# 获取点的数量
def get_point_num(path):
    file_list = [f for f in os.listdir(path) if f.endswith("_resampled_coordinates.txt")]
    sample_num = len(file_list)

    samples = []
    for i in range(sample_num):
        txt_path = os.path.join(path, f"{i + 1}_resampled_coordinates.txt")
        with open(txt_path, "r") as f:
            ori_data = f.read().strip().split("\n")
            samples.append(len(ori_data))  # 计算行数作为点的数量
    return np.array(samples, dtype=int)


def pingcha(X_data, X_text, point_num):
    res = []
    for i in range(X_data.shape[0]):
        data = X_data[i, 0]
        pnum = int(point_num[i]) - 1

        det_x = [float(data[2 * j]) for j in range(pnum)]
        det_y = [float(data[2 * j + 1]) for j in range(pnum)]

        point_x, point_y = 0.0, 0.0
        x, y = [point_x], [point_y]
        for j in range(pnum):
            point_x += det_x[j]
            point_y += det_y[j]
            x.append(point_x)
            y.append(point_y)

        data2 = X_text[i, 0]
        det_x2 = [float(data2[2 * j]) for j in range(pnum)]
        det_y2 = [float(data2[2 * j + 1]) for j in range(pnum)]

        point_x2, point_y2 = 0.0, 0.0
        x2, y2 = [point_x2], [point_y2]
        for j in range(pnum):
            point_x2 += det_x2[j]
            point_y2 += det_y2[j]
            x2.append(point_x2)
            y2.append(point_y2)

        sum_det_x = x[0] - x[pnum] - (x2[0] - x2[pnum])
        sum_det_y = y[0] - y[pnum] - (y2[0] - y2[pnum])

        if pnum - 1 == 0:
            res_data = data
        else:
            x_revise = sum_det_x / (pnum)
            y_revise = sum_det_y / (pnum)
            det_x_res = [det_x[j] + x_revise for j in range(pnum)]
            det_y_res = [det_y[j] + y_revise for j in range(pnum)]

            res_data = []
            for j in range(pnum):
                res_data.append(det_x_res[j])
                res_data.append(det_y_res[j])

        res_data = np.array(res_data).reshape(1, 200)
        if len(res) == 0:
            res = res_data[np.newaxis, :, :]
        else:
            res = np.concatenate((res, res_data[np.newaxis, :, :]), axis=0)
    return res


def draw_det(data, point_num):
    det_x = [float(data[2 * i]) for i in range(int(point_num) - 1)]
    det_y = [float(data[2 * i + 1]) for i in range(int(point_num) - 1)]

    point_x = 0.0
    point_y = 0.0
    x = [point_x]
    y = [point_y]

    for i in range(int(point_num) - 1):
        point_x += det_x[i]
        point_y += det_y[i]
        x.append(point_x)
        y.append(point_y)

    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    return x, y, xmin, xmax, ymin, ymax


def arcgis(data, start_point):
    det_x = [float(data[2 * i]) for i in range(100)]
    det_y = [float(data[2 * i + 1]) for i in range(100)]

    start_point_x = str(start_point).split(",")[0]
    start_point_y = str(start_point).split(",")[1]
    point_x = float(start_point_x)
    point_y = float(start_point_y)

    x = [point_x]
    y = [point_y]

    for i in range(100):
        point_x += det_x[i]
        point_y += det_y[i]
        x.append(point_x)
        y.append(point_y)
    return x, y


# ==========================================
# 3. 计算几何与拓扑指标函数
# ==========================================
def calculate_polygon_metrics(x_orig, y_orig, x_recon, y_recon):
    """计算闭合多边形的相对面积误差、相对周长误差和 IoU"""
    poly_P = Polygon(zip(x_orig, y_orig))
    poly_Q = Polygon(zip(x_recon, y_recon))

    # 修复可能的自相交多边形
    if not poly_P.is_valid:
        poly_P = poly_P.buffer(0)
    if not poly_Q.is_valid:
        poly_Q = poly_Q.buffer(0)

    A_P = poly_P.area
    A_Q = poly_Q.area
    delta_A = abs(A_P - A_Q) / A_P if A_P > 0 else 0.0

    L_P = poly_P.length
    L_Q = poly_Q.length
    delta_L = abs(L_P - L_Q) / L_P if L_P > 0 else 0.0

    try:
        intersection = poly_P.intersection(poly_Q).area
        union = poly_P.union(poly_Q).area
        iou = intersection / union if union > 0 else 0.0
    except Exception:
        iou = 0.0

    return delta_A, delta_L, iou


def calculate_curvature_error(x_orig, y_orig, x_recon, y_recon):
    """计算曲率变化误差 (Menger 曲率)"""

    def get_curvature(x_coords, y_coords):
        if x_coords[0] == x_coords[-1] and y_coords[0] == y_coords[-1]:
            x_coords, y_coords = x_coords[:-1], y_coords[:-1]

        n = len(x_coords)
        curvatures = np.zeros(n)

        for i in range(n):
            prev_i = (i - 1) % n
            next_i = (i + 1) % n

            v1_x = x_coords[i] - x_coords[prev_i]
            v1_y = y_coords[i] - y_coords[prev_i]
            v2_x = x_coords[next_i] - x_coords[i]
            v2_y = y_coords[next_i] - y_coords[i]

            norm_v1 = np.sqrt(v1_x ** 2 + v1_y ** 2)
            norm_v2 = np.sqrt(v2_x ** 2 + v2_y ** 2)

            if norm_v1 == 0 or norm_v2 == 0:
                continue

            cross_prod_mag = abs(v1_x * v2_y - v1_y * v2_x)
            sin_theta = cross_prod_mag / (norm_v1 * norm_v2)
            kappa = (2 * sin_theta) / (norm_v1 + norm_v2)
            curvatures[i] = kappa

        return curvatures

    kappa_P = get_curvature(x_orig, y_orig)
    kappa_Q = get_curvature(x_recon, y_recon)

    min_len = min(len(kappa_P), len(kappa_Q))
    if min_len == 0:
        return 0.0
    delta_kappa = np.mean(np.abs(kappa_P[:min_len] - kappa_Q[:min_len]))

    return delta_kappa


# ==========================================
# 4. 主程序：加载模型与动态路径配置
# ==========================================
name = "CAE_200.pth"
mymodel = torch.load(name, weights_only=False).to(device)

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 data 文件夹路径（向上两级）
data_dir = os.path.join(current_dir, '..', '..', 'data')

# 数据读取路径
det_path = os.path.join(data_dir, 'det_topo_part')
val_data = load_txt_files(det_path)

# 转置
val_data = val_data.transpose(0, 2, 1)
print(f"数据形状: {val_data.shape}")

# 标准化 均值0，方差1
mean_val = val_data.mean(axis=(1, 2), keepdims=True)
std_val = val_data.std(axis=(1, 2), keepdims=True) + 1e-8
val_data_normalized = (val_data - mean_val) / std_val

# 转化为张量并移动到GPU
val_data_tensor = torch.tensor(val_data_normalized).float().to(device)

# 模型推理
startTime = time.time()
decoded_val = mymodel(val_data_tensor)
endtime = time.time()
print(f"推理耗时: {endtime - startTime:.4f} 秒")

# 标准化还原
restored_val = decoded_val.detach().cpu().numpy() * std_val + mean_val
val_data = val_data_tensor.detach().cpu().numpy() * std_val + mean_val

point_path = os.path.join(data_dir, 'point_topo_part')
start_point = get_start_point(point_path)
last_point = get_last_point(point_path)
point_num = get_point_num(point_path)

if len(point_num) == 0:
    raise ValueError("未能读取到 point_num，请检查 point_path 路径下是否存在正确的文件名格式！")

# 平差处理
restored_val = pingcha(restored_val, val_data, point_num)

# 写入文件路径配置
restored_det_path = os.path.join(data_dir, 'decode_det2')
restored_point_path = os.path.join(data_dir, 'decode_point2')

# 确保输出目录存在
os.makedirs(restored_det_path, exist_ok=True)
os.makedirs(restored_point_path, exist_ok=True)

for i in range(start_point.shape[0]):
    x2, y2 = arcgis(restored_val[i, 0], start_point[i])
    point_file_path = os.path.join(restored_point_path, f"{i}_point.txt")
    det_file_path = os.path.join(restored_det_path, f"{i}_det.txt")

    with open(point_file_path, "w") as f_point:
        for j in range(len(x2)):
            f_point.write(f"{x2[j]},{y2[j]}\n")

    with open(det_file_path, "w") as f_det:
        f_det.write(",".join(map(str, restored_val[i])) + ",")

# ==========================================
# 5. 点级别误差评估
# ==========================================
max_res, min_res, avg_res, median, std = [], [], [], [], []

for i in range(restored_val.shape[0]):
    x, y, xmin, xmax, ymin, ymax = draw_det(val_data[i, 0], point_num[i])
    x2, y2, xmin2, xmax2, ymin2, ymax2 = draw_det(restored_val[i, 0], point_num[i])

    # 原有误差计算 (替换了 np.mat 为 np.asmatrix 避免高版本警告)
    x_mat, y_mat = np.asmatrix(x), np.asmatrix(y)
    x2_mat, y2_mat = np.asmatrix(x2), np.asmatrix(y2)
    detx = x2_mat - x_mat
    dety = y2_mat - y_mat

    detpoint = np.sqrt(np.multiply(detx, detx) + np.multiply(dety, dety))
    real_detpoint = detpoint[:, 1:-1]

    if real_detpoint.shape[1] != 0:
        max_index = np.unravel_index(np.argmax(real_detpoint, axis=None), real_detpoint.shape)
        max_res.append(real_detpoint[max_index])

        min_index = np.unravel_index(np.argmin(real_detpoint, axis=None), real_detpoint.shape)
        min_res.append(real_detpoint[min_index])

        real_detpoint_arry = real_detpoint.getA()
        median.append(np.median(real_detpoint_arry[0]))
        std.append(np.std(real_detpoint_arry[0]))
        avg_res.append(np.sum(real_detpoint) / real_detpoint.shape[1])

# 原有统计指标加权平均
avg_det, avg_mediem, avg_std = 0, 0, 0
point_num_sum = sum([int(p) - 2 for p in point_num])

max_det = max(max_res) if max_res else 0
for i in range(len(avg_res)):
    if int(point_num[i]) > 2:
        weight = (int(point_num[i]) - 2) / point_num_sum
        avg_det += avg_res[i] * weight
        avg_mediem += median[i] * weight
        avg_std += std[i] * weight

print("\n--- 原始点级别评价指标 ---")
print(f"最大偏移距离: {max_det}")
print(f"平均偏移距离: {avg_det}")
print(f"平均偏移距离中位数: {avg_mediem}")
print(f"平均偏移距离标准差: {avg_std}")

# ==========================================
# 6. 新增：拼接分块线段为完整岛屿，并计算几何/拓扑指标
# ==========================================
print("\n正在拼接分块数据以重建完整闭合岛屿...")

orig_islands_x, orig_islands_y = [], []
recon_islands_x, recon_islands_y = [], []
curr_ox, curr_oy = [], []
curr_rx, curr_ry = [], []

for i in range(val_data.shape[0]):
    ox, oy = arcgis(val_data[i, 0], start_point[i])
    rx, ry = arcgis(restored_val[i, 0], start_point[i])

    if len(curr_ox) == 0:
        curr_ox.extend(ox)
        curr_oy.extend(oy)
        curr_rx.extend(rx)
        curr_ry.extend(ry)
    else:
        # 判定距离：如果当前片段的起点距离上一个片段的终点很近(容差 < 1e-2)，则拼接
        dist = math.hypot(ox[0] - curr_ox[-1], oy[0] - curr_oy[-1])
        if dist < 1e-2:
            curr_ox.extend(ox[1:])
            curr_oy.extend(oy[1:])
            curr_rx.extend(rx[1:])
            curr_ry.extend(ry[1:])
        else:
            orig_islands_x.append(curr_ox)
            orig_islands_y.append(curr_oy)
            recon_islands_x.append(curr_rx)
            recon_islands_y.append(curr_ry)

            curr_ox, curr_oy = list(ox), list(oy)
            curr_rx, curr_ry = list(rx), list(ry)

if len(curr_ox) > 0:
    orig_islands_x.append(curr_ox)
    orig_islands_y.append(curr_oy)
    recon_islands_x.append(curr_rx)
    recon_islands_y.append(curr_ry)

island_count = len(orig_islands_x)
print(f"✅ 成功拼接出 {island_count} 个完整的岛屿！")

area_errors, perimeter_errors, iou_scores, curvature_errors = [], [], [], []

for idx in range(island_count):
    ox, oy = orig_islands_x[idx], orig_islands_y[idx]
    rx, ry = recon_islands_x[idx], recon_islands_y[idx]

    delta_A, delta_L, iou = calculate_polygon_metrics(ox, oy, rx, ry)
    delta_kappa = calculate_curvature_error(ox, oy, rx, ry)

    area_errors.append(delta_A)
    perimeter_errors.append(delta_L)
    iou_scores.append(iou)
    curvature_errors.append(delta_kappa)

print("\n--- 岛屿级几何与拓扑指标 ---")
print(f"平均相对面积误差: {np.mean(area_errors) * 100:.4f}%")
print(f"平均相对周长误差: {np.mean(perimeter_errors) * 100:.4f}%")
print(f"平均 IoU (交并比): {np.mean(iou_scores):.4f}")
print(f"平均曲率变化误差: {np.mean(curvature_errors):.6f}")

# ==========================================
# 7. 数据导出至 CSV 与 Excel
# ==========================================
print("\n正在将评估指标导出为表格...")

# 统一输出到指定的 output 文件夹 (根据你提供的路径)
output_dir = r"C:\Users\Administrator\Desktop\TGIS投稿\Data and Code\output"
os.makedirs(output_dir, exist_ok=True)

metrics_data = {
    "Island_ID": range(1, island_count + 1),
    "Relative_Area_Error": area_errors,
    "Relative_Perimeter_Error": perimeter_errors,
    "IoU": iou_scores,
    "Curvature_Error": curvature_errors
}

df_metrics = pd.DataFrame(metrics_data)
csv_save_path = os.path.join(output_dir, "evaluation_metrics.csv")
excel_save_path = os.path.join(output_dir, "evaluation_metrics.xlsx")

df_metrics.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
df_metrics.to_excel(excel_save_path, index=False)

print(f"✅ 数据导出成功！")
print(f"CSV 路径: {csv_save_path}")
print(f"Excel 路径: {excel_save_path}")