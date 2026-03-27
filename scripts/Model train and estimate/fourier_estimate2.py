import math
import os
import numpy as np
import time
from shapely.geometry import Polygon
import pandas as pd


# ==========================================
# 1. 傅里叶变换核心类 (向量化优化版)
# ==========================================
class Fourier:
    def __init__(self, sample, point_num):
        x, y, xmin, xmax, ymin, ymax, length, sum_length, area, triangle = draw_det(sample, point_num)
        self.ori_x = np.array(x)
        self.ori_y = np.array(y)

        mx, my, mlength = self.mirror(list(x), list(y), list(length))
        self.x = np.array(mx)
        self.y = np.array(my)
        self.length = np.array(mlength)

        self.sum_length = 2 * sum_length
        self.s = np.zeros(len(self.length) + 1)
        self.s[1:] = np.cumsum(self.length)

    def mirror(self, x, y, length):
        if len(x) <= 2: return x, y, length
        mx, my, mlen = list(x), list(y), list(length)
        center_x = (x[0] + x[-1]) / 2
        center_y = (y[0] + y[-1]) / 2
        for i in range(len(x) - 1):
            mx.append(2 * center_x - x[i + 1])
            my.append(2 * center_y - y[i + 1])
            mlen.append(length[i])
        return mx, my, mlen

    def _calc_ratio_vectorized(self, coords, item_num, is_cos=True):
        """完全向量化的系数计算，消除 Python 层级的双重 for 循环"""
        N = len(self.length)
        s_i = self.s[:-1]
        s_next = self.s[1:]
        delta_s = s_next - s_i

        # 避免除以 0
        valid_idx = delta_s > 1e-8
        d_chu_s = np.zeros(N)
        d_chu_s[valid_idx] = (coords[1:] - coords[:-1])[valid_idx] / delta_s[valid_idx]

        n_arr = np.arange(1, item_num + 1)
        l_chu_2npi = self.sum_length / (2 * n_arr * np.pi)

        # 扩展维度以进行矩阵运算 (项数, 数据点数)
        l_chu_2npi = l_chu_2npi[:, np.newaxis]
        n_arr = n_arr[:, np.newaxis]

        s_next_scaled = s_next / l_chu_2npi
        s_i_scaled = s_i / l_chu_2npi

        term1 = coords[:-1] - d_chu_s * s_i

        if is_cos:
            p1 = term1 * l_chu_2npi * (np.sin(s_next_scaled) - np.sin(s_i_scaled))
            p2 = d_chu_s * (l_chu_2npi ** 2) * (
                        s_next_scaled * np.sin(s_next_scaled) - s_i_scaled * np.sin(s_i_scaled) + np.cos(
                    s_next_scaled) - np.cos(s_i_scaled))
        else:
            p1 = term1 * l_chu_2npi * (-1) * (np.cos(s_next_scaled) - np.cos(s_i_scaled))
            p2 = d_chu_s * (l_chu_2npi ** 2) * (np.sin(s_next_scaled) - np.sin(s_i_scaled) - (
                        s_next_scaled * np.cos(s_next_scaled) - s_i_scaled * np.cos(s_i_scaled)))

        sum_val = np.sum(p1 + p2, axis=1)
        return (2 / self.sum_length) * sum_val

    def get_F_line(self, item_num, resp_num):
        gap = self.sum_length / (resp_num - 1)

        # 向量化计算 A0
        delta_s = self.s[1:] - self.s[:-1]
        valid_idx = delta_s > 1e-8

        d_chu_s_x = np.zeros(len(delta_s))
        d_chu_s_y = np.zeros(len(delta_s))
        d_chu_s_x[valid_idx] = (self.x[1:] - self.x[:-1])[valid_idx] / delta_s[valid_idx]
        d_chu_s_y[valid_idx] = (self.y[1:] - self.y[:-1])[valid_idx] / delta_s[valid_idx]

        a0x = (2 / self.sum_length) * np.sum(
            0.5 * d_chu_s_x * (self.s[1:] ** 2 - self.s[:-1] ** 2) + (self.x[:-1] - d_chu_s_x * self.s[:-1]) * delta_s)
        a0y = (2 / self.sum_length) * np.sum(
            0.5 * d_chu_s_y * (self.s[1:] ** 2 - self.s[:-1] ** 2) + (self.y[:-1] - d_chu_s_y * self.s[:-1]) * delta_s)

        # 一次性算出所有项的系数
        anx = self._calc_ratio_vectorized(self.x, item_num, True)
        bnx = self._calc_ratio_vectorized(self.x, item_num, False)
        any_coeff = self._calc_ratio_vectorized(self.y, item_num, True)
        bny = self._calc_ratio_vectorized(self.y, item_num, False)

        # 向量化生成重构线
        s_vals = np.arange(resp_num) * gap
        n_arr = np.arange(1, item_num + 1)[:, np.newaxis]
        angles = 2 * n_arr * np.pi * s_vals / self.sum_length

        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        x_res = 0.5 * a0x + np.dot(anx, cos_angles) + np.dot(bnx, sin_angles)
        y_res = 0.5 * a0y + np.dot(any_coeff, cos_angles) + np.dot(bny, sin_angles)

        return x_res.tolist(), y_res.tolist()


# ==========================================
# 2. 几何评估函数
# ==========================================
def calculate_metrics(x_orig, y_orig, x_recon, y_recon):
    half = len(x_recon) // 2
    xr, yr = x_recon[:half + 1], y_recon[:half + 1]

    poly_p = Polygon(zip(x_orig, y_orig))
    poly_q = Polygon(zip(xr, yr))

    if not poly_p.is_valid: poly_p = poly_p.buffer(0)
    if not poly_q.is_valid: poly_q = poly_q.buffer(0)

    area_err = abs(poly_p.area - poly_q.area) / poly_p.area if poly_p.area > 0 else 0
    peri_err = abs(poly_p.length - poly_q.length) / poly_p.length if poly_p.length > 0 else 0
    iou = poly_p.intersection(poly_q).area / poly_p.union(poly_q).area if poly_p.union(poly_q).area > 0 else 0

    def get_curvature(xc, yc):
        xc, yc = np.array(xc), np.array(yc)
        v1x, v1y = xc[1:-1] - xc[:-2], yc[1:-1] - yc[:-2]
        v2x, v2y = xc[2:] - xc[1:-1], yc[2:] - yc[1:-1]
        n1, n2 = np.hypot(v1x, v1y), np.hypot(v2x, v2y)

        valid = (n1 > 0) & (n2 > 0)
        curv = np.zeros(len(xc))
        sin_t = np.abs(v1x[valid] * v2y[valid] - v1y[valid] * v2x[valid]) / (n1[valid] * n2[valid])
        curv[1:-1][valid] = (2 * sin_t) / (n1[valid] + n2[valid])
        return curv

    k_p, k_q = get_curvature(x_orig, y_orig), get_curvature(xr, yr)
    ml = min(len(k_p), len(k_q))
    curv_err = np.mean(np.abs(k_p[:ml] - k_q[:ml])) if ml > 0 else 0

    return area_err, peri_err, iou, curv_err


# ==========================================
# 3. 辅助读取函数
# ==========================================
def get_point_num(path):
    file_list = [f for f in os.listdir(path) if f.endswith("_resampled_coordinates.txt") and not f.startswith("diff_")]
    samples = []
    for i in range(len(file_list)):
        txt_path = os.path.join(path, f"{i + 1}_resampled_coordinates.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                ori_data = f.read().strip().split("\n")
                samples.append(len(ori_data))
    return np.array(samples, dtype=int)


def get_sample(folder_path):
    file_list = [f for f in os.listdir(folder_path) if f.startswith("diff_")]
    all_data = []
    for i in range(len(file_list)):
        file_name = f"diff_{i + 1}_resampled_coordinates.txt"
        file_path = os.path.join(folder_path, file_name)
        data = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                for line in file:
                    values = list(map(float, line.strip().split(',')))
                    data.extend(values)
            if len(data) == 200:
                all_data.append(np.array(data).reshape(200, 1))
    return np.stack(all_data)


def draw_det(data, point_num):
    det_x = [float(data[2 * i][0]) for i in range(int(point_num) - 1)]
    det_y = [float(data[2 * i + 1][0]) for i in range(int(point_num) - 1)]
    x, y, length = [0.0], [0.0], []
    sl, area = 0.0, 0.0
    for i in range(len(det_x)):
        l = math.sqrt(det_x[i] ** 2 + det_y[i] ** 2)
        x.append(x[-1] + det_x[i])
        y.append(y[-1] + det_y[i])
        length.append(l)
        sl += l
        area += (det_x[i] * det_y[i]) / 2
    return x, y, min(x), max(x), min(y), max(y), length, sl, area, None


# ==========================================
# 4. 执行推理与打印指标
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))

det_path = os.path.join(data_dir, 'det2')
point_path = os.path.join(data_dir, 'point2')

x_train = get_sample(det_path)
p_nums = get_point_num(point_path)

if len(x_train) == 0 or len(p_nums) == 0:
    print("错误：未找到数据，请检查路径。")
    exit()

# 数据截断，只处理前 18 个片段（对应一个完整的岛屿集）
x_train = x_train[:18]
p_nums = p_nums[:18]

results = []
start_t = time.time()

print(f"正在计算傅里叶重构指标 (共 {len(x_train)} 条数据)...")
for i in range(x_train.shape[0]):
    f_model = Fourier(x_train[i], p_nums[i])
    x_f, y_f = f_model.get_F_line(100, int(p_nums[i]) * 2 - 1)
    ae, pe, iou, ce = calculate_metrics(f_model.ori_x, f_model.ori_y, x_f, y_f)
    results.append([i + 1, ae, pe, iou, ce])

end_t = time.time()

df = pd.DataFrame(results, columns=['ID', 'Area_Error', 'Peri_Error', 'IoU', 'Curv_Error'])

print("\n" + "=" * 40)
print("       傅里叶重构评估报告")
print("=" * 40)
print(f"平均相对面积误差: {df['Area_Error'].mean() * 100:.6f} %")
print(f"平均相对周长误差: {df['Peri_Error'].mean() * 100:.6f} %")
print(f"平均 IoU (交并比): {df['IoU'].mean():.6f}")
print(f"平均曲率变化误差: {df['Curv_Error'].mean():.8f}")
print("=" * 40)

out_path = os.path.join(os.path.dirname(data_dir), "output", "fourier_evaluation.xlsx")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_excel(out_path, index=False)
print(f"详细报告已导出至: {out_path}")
