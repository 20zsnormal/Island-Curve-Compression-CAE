import math
import torch
from torch import nn
from torch.nn import Sequential,Conv1d,MSELoss,Tanh,BatchNorm1d,ReLU
import os
import numpy as np
from sklearn.model_selection import train_test_split
import time


device=torch.device("cuda")
class Encode_Decode(nn.Module):
    def __init__(self,encoded_space_dim=100):
        super(Encode_Decode, self).__init__()
        # Encoder
        self.Encode = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=5, padding=1),
            #nn.Conv1d(in_channels=1, out_channels=32, kernel_size=10, stride=2, padding=0),

            nn.BatchNorm1d(16),  # 添加批量归一化层
            nn.ReLU(True),
            # nn.Tanh(),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=5, padding=1),
            #nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10, stride=2, padding=0),

            nn.BatchNorm1d(32),  # 添加批量归一化层
            nn.ReLU(True),
            # nn.Tanh()
        )
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear p
        self.encoder_lin = nn.Sequential(
            nn.Linear( 32 * 8, encoded_space_dim),
            #nn.Linear( 64 * 94, 512),
            # nn.Tanh(),
            nn.ReLU(True),
            # nn.Linear(256, encoded_space_dim),
            # nn.Tanh(),
        )

        # Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 32 * 8),
            #nn.Linear(512, 64 * 94),
            nn.ReLU(True),
            # nn.Tanh(),

            # nn.Linear(256, 64 * 8),
            # nn.Tanh(),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 8))
        #self.unflatten = nn.Unflatten(dim=1,unflattened_size=(64, 94))
        self.Decode = nn.Sequential(

            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=7, stride=5, padding=1, output_padding=0),
            #nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=10, stride=2, padding=0, output_padding=0),

            nn.BatchNorm1d(16),  # 添加批量归一化层
            nn.ReLU(True),
            # nn.Tanh(),

            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=7, stride=5, padding=1, output_padding=0),
            #nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=10, stride=2, padding=0, output_padding=0),
            # nn.Sigmoid()
        )
    def forward(self,x):
        x=self.Encode(x)
        x=self.flatten(x)
        compress=self.encoder_lin(x)
        x=self.decoder_lin(compress)
        x=self.unflatten(x)
        x=self.Decode(x)
        return x
# class Encode_Decode(nn.Module):
#
#     def __init__(self,encoded_space_dim=100):
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
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(f'Selected device: {device}')

#保存的参数
name="C:\\Users\\ZS-1PC\\Desktop\\数据与代码\\海洋面All Step Data\\岛群\\parmspath\\60\\CAE_186.pth"
# name="等高线ex\\DGyuYSB_"+str(length)+".h5"

#载入训练的数据
mymodel=torch.load(name)
mymodel=mymodel.to(device)
# 定义读取文件夹中的txt文件，并将数据转换为numpy数组的函数
def load_txt_files(folder_path):
    file_list = os.listdir(folder_path)
    sample_num = len(file_list)
    all_data = []  # 存储所有文件的数据
    # 遍历文件夹中的每个文件
    for i in range(int(sample_num)):
        file_name="diff_" + str(i+1) + "_resampled_coordinates.txt"
        #file_name=str(i+1)+"_resampled_coordinates.txt"
        file_path = folder_path + "\\"+file_name
        # 读取文件内容并将其拆分为100行2列的浮点数列表
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                values = list(map(float, line.strip().split(',')))  # 将每一行的数据转换为浮点数
                data.extend(values)  # 将2列展开为1列
        # 确保每个文件有200个数据点
        # if len(data) == 400:
        #     reshaped_data = np.array(data).reshape(400, 1)  # 将其转换为400x1
        #     all_data.append(reshaped_data)
        #文件有100个数据点
        if len(data) == 200:
            reshaped_data = np.array(data).reshape(200, 1)  # 将其转换为400x1
            all_data.append(reshaped_data)
        else:
            print(f"文件 {file_name} 的数据长度不是400，跳过该文件。")
    # 将所有文件的数据堆叠在一起，形成一个三维矩阵 (文件数, 400, 1)
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
#平差
def pingcha(X_data, X_text, point_num):
    """
    平差函数，用于修正解码后的数据以匹配原始点分布。

    参数:
    - X_data: 解码后的数据，形状 (样本数, 1, 400)。
    - X_text: 编码前的原始数据，形状 (样本数, 1, 400)。
    - point_num: 每个样本的点数列表，形状 (样本数,)。

    返回:
    - res: 修正后的数据，形状 (样本数, 1, 400)。
    """
    # 初始化修正后的结果数组
    res = []

    for i in range(X_data.shape[0]):
        # 编码解码后的数据处理
        data = X_data[i, 0]  # 提取当前样本数据 (400,)
        pnum = int(point_num[i])-1  # 当前样本的点数
        #pnum = int(point_num[i])   # 当前样本的点数

        det_x = [float(data[2 * j]) for j in range(pnum)]
        det_y = [float(data[2 * j + 1]) for j in range(pnum)]

        # 初始化累积坐标
        point_x, point_y = 0.0, 0.0
        x, y = [point_x], [point_y]
        for j in range(pnum):
            point_x += det_x[j]
            point_y += det_y[j]
            x.append(point_x)
            y.append(point_y)

        # 编码前的原始数据处理
        data2 = X_text[i, 0]  # 提取当前样本数据 (400,)
        det_x2 = [float(data2[2 * j]) for j in range(pnum)]
        det_y2 = [float(data2[2 * j + 1]) for j in range(pnum)]

        point_x2, point_y2 = 0.0, 0.0
        x2, y2 = [point_x2], [point_y2]
        for j in range(pnum):
            point_x2 += det_x2[j]
            point_y2 += det_y2[j]
            x2.append(point_x2)
            y2.append(point_y2)

        # 计算差值
        sum_det_x = x[0] - x[pnum ] - (x2[0] - x2[pnum])
        sum_det_y = y[0] - y[pnum ] - (y2[0] - y2[pnum])

        # 平均分配差值
        if pnum - 1 == 0:
            res_data = data  # 如果只有一个点，不需要修正
        else:
            x_revise = sum_det_x / (pnum)
            y_revise = sum_det_y / (pnum)

            # 修正后的增量值
            det_x_res = [det_x[j] + x_revise for j in range(pnum)]
            det_y_res = [det_y[j] + y_revise for j in range(pnum)]

            # 将修正后的增量值重新排列
            res_data = []
            for j in range(pnum):
                res_data.append(det_x_res[j])
                res_data.append(det_y_res[j])

        # 转换为与输入相同的形状 (1, 400)
        #res_data = np.array(res_data).reshape(1, 400)
        # 转换为与输入相同的形状 (1, 200)
        res_data = np.array(res_data).reshape(1, 200)
        # 累积结果
        if len(res) == 0:
            res = res_data[np.newaxis, :, :]
        else:
            res = np.concatenate((res, res_data[np.newaxis, :, :]), axis=0)

    return res

def draw_det(data,point_num):

    det_x = [float(data[2 * i]) for i in range(int(point_num)-1)]
    det_y = [float(data[2 * i + 1]) for i in range(int(point_num)-1)]

    point_x = 0.0
    point_y = 0.0
    x = []
    y = []
    x.append(point_x)
    y.append(point_y)

    for i in range(int(point_num)-1):
        point_x += det_x[i]
        point_y += det_y[i]
        x.append(point_x)
        y.append(point_y)
    # x.append(0.0)
    # y.append(0.0)

    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)

    return x, y, xmin, xmax, ymin, ymax

# 使用该函数读取文件夹中的txt文件并转换
det_path = "C:\\Users\\ZS-1PC\\Desktop\\数据与代码\\海洋面All Step Data\\岛群\\det_topo_part"   # 请替换为你的文件夹路径

val_data = load_txt_files(det_path)
#print("最终的三维矩阵为:", input_array)
#转置
val_data=val_data.transpose(0,2,1)
print(val_data.shape)
#标准化 均值0，方差1
mean_val = val_data.mean(axis=(1, 2), keepdims=True)
std_val = val_data.std(axis=(1, 2), keepdims=True) + 1e-8
val_data_normalized = (val_data - mean_val) / std_val

# 归一化到 [0, 1]
# min_val = val_data.min(axis=(1, 2), keepdims=True)  # 按样本计算最小值
# max_val = val_data.max(axis=(1, 2), keepdims=True)  # 按样本计算最大值
# val_data_normalized = (val_data - min_val) / (max_val - min_val + 1e-8)  # 防止除以零


# 转化为张量并移动到GPU
val_data_tensor = torch.tensor(val_data_normalized).float().cuda()

# 模型推理
startTime = time.time()
decoded_val = mymodel(val_data_tensor)
endtime = time.time()
diffrentTime = endtime - startTime
print(diffrentTime)

#标准化还原
restored_val = decoded_val.detach().cpu().numpy() * std_val + mean_val

val_data=val_data_tensor.detach().cpu().numpy() * std_val + mean_val
#归一化还原（0-1）
# restored_val = decoded_val.detach().cpu().numpy() * (max_val - min_val) + min_val
#
# val_data=val_data_tensor.detach().cpu().numpy() * (max_val - min_val) + min_val
#归一化还原(-1-1)
# restored_val = (decoded_val.detach().cpu().numpy()/2+0.5) * (max_val - min_val) + min_val
#
# val_data=(val_data.detach().cpu().numpy()/2+0.5) * (max_val - min_val) + min_val

# r=2
# fig, axs = plt.subplots(r, 2)
# for i in range(r):
#     id=random.randint(0,X_test.shape[0]-1)
#     x, y, xmin, xmax, ymin, ymax = draw_det(X_test[id])
#     axs[i, 0].plot(x, y)
#     axs[i, 0].axis([xmin, xmax, ymin, ymax])
#
#     x, y, xmin, xmax, ymin, ymax = draw_det(decoded_imgs[id])
#     axs[i, 1].plot(x, y)
#     axs[i, 1].axis([xmin, xmax, ymin, ymax])
# plt.show()

def fmax(a,b):
    return a if a > b else b

def fmin(a,b):
    return a if a < b else b

# 显示分类效果

# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# r, c = 2, 2
# fig, axs = plt.subplots(r, c)
# for i in range(r):
#     for j in range(c):
#         id=random.randint(0,X_test.shape[0]-1)
#         x, y, xmin, xmax, ymin, ymax = draw_det(X_test[id])
#         x2, y2, xmin2, xmax2, ymin2, ymax2 = draw_det(decoded_imgs[id])
#         axs[i, j].plot(x, y, label='原始线要素')
#         axs[i, j].plot(x2, y2, label='解码后线要素')
#         xmin = fmin(xmin,xmin2)
#         xmax = fmax(xmax, xmax2)
#         ymin = fmin(ymin, ymin2)
#         ymax = fmax(ymax, ymax2)
#         axs[i, j].axis([xmin, xmax, ymin, ymax])
#
#         plt.tight_layout()
#
#
# plt.show()

#输出结果
def arcgis(data,start_point):
    #根据实际采样点进行改变
    det_x = [float(data[2 * i]) for i in range(100)]
    det_y = [float(data[2 * i + 1]) for i in range(100)]

    start_point_x = str(start_point).split(",")[0]
    start_point_y = str(start_point).split(",")[1]
    point_x = float(start_point_x)
    point_y = float(start_point_y)
    x = []
    y = []
    x.append(point_x)
    y.append(point_y)
    #根据实际采样点进行改变
    for i in range(100):
        point_x += det_x[i]
        point_y += det_y[i]
        x.append(point_x)
        y.append(point_y)
    return x,y

point_path = "C:\\Users\\ZS-1PC\\Desktop\\数据与代码\\海洋面All Step Data\\岛群\\point_topo_part"   # 请替换为你的文件夹路径
#计算第一个点、最后一个点、点数量
start_point=get_start_point(point_path)
last_point=get_last_point(point_path)
point_num=get_point_num(point_path)
print(point_num)
restored_det_path = "C:\\Users\\ZS-1PC\\Desktop\\数据与代码\\海洋面All Step Data\\岛群\\decode_det2"   # 请替换为你的文件夹路径
restored_point_path = "C:\\Users\\ZS-1PC\\Desktop\\数据与代码\\海洋面All Step Data\\岛群\\decode_point2"   # 请替换为你的文件夹路径
#平差
restored_val=pingcha(restored_val,val_data,point_num)
for i in range(start_point.shape[0]):
    # 调用 arcgis 函数获取 x2 和 y2
    x2, y2 = arcgis(restored_val[i,0], start_point[i])
    # 生成文件路径
    point_file_path = os.path.join(restored_point_path, f"{i}_point.txt")
    det_file_path = os.path.join(restored_det_path, f"{i}_det.txt")

    # 写入 _point.txt 文件
    with open(point_file_path, "w") as f_point:
        for j in range(len(x2)):
            f_point.write(f"{x2[j]},{y2[j]}\n")  # 写入坐标
    # 写入 _det.txt 文件
    with open(det_file_path, "w") as f_det:
        f_det.write(",".join(map(str, restored_val[i])) + ",")  # 写入 det 数据

# for i in range(start_point.shape[0]):
#     # 生成文件路径
#     point_file_path = os.path.join(restored_point_path, f"{i}_point.txt")
#     # 写入 _point.txt 文件
#     with open(point_file_path, "w") as f_point:
#         f_point.write(",".join(map(str, restored_val[i])) + ",")  # 写入坐标

# regien 计算偏移量(平差)
max_res=[]
min_res=[]
avg_res=[]
median=[]
std=[]
max_det=0
mim_det=0
avg_det=0
avg_mediem=0
avg_std=0
for i in range(restored_val.shape[0]):
    x, y, xmin, xmax, ymin, ymax = draw_det(val_data[i,0],point_num[i])
    x2, y2, xmin2, xmax2, ymin2, ymax2 = draw_det(restored_val[i,0],point_num[i])
    x=np.mat(x)
    y = np.mat(y)
    x2 = np.mat(x2)
    y2 = np.mat(y2)
    detx=x2-x
    dety=y2-y

    detpoint=np.sqrt(np.multiply(detx,detx)+np.multiply(dety,dety))
    real_detpoint=detpoint[:,1:-1]

    if  (real_detpoint.shape[1]!=0):
        # numpy最大值坐标
        max_index = np.unravel_index(np.argmax(real_detpoint, axis=None), real_detpoint.shape)
        # numpy最大值
        max_res.append(real_detpoint[max_index])
        # numpy最x小值坐标
        min_index = np.unravel_index(np.argmin(real_detpoint, axis=None), real_detpoint.shape)
        min_res.append(real_detpoint[min_index])
        # 中位数
        real_detpoint_arry=real_detpoint.getA()

        median.append(np.median(real_detpoint_arry[0]))
        std.append(np.std(real_detpoint_arry[0]))
        # print(np.median(real_detpoint[0]))
        avg_res.append(np.sum(real_detpoint)/real_detpoint.shape[1])

point_num_sum=0
for i in range(val_data.shape[0]):
    point_num_sum+=int(point_num[i])-2
max_det=max(max_res)
min_det=min(min_res)
for i in range(len(avg_res)):
    if  int(point_num[i])>2:
        avg_det+=avg_res[i]*(int(point_num[i])-2)/point_num_sum
        avg_mediem+=median[i]*(int(point_num[i])-2)/point_num_sum
        avg_std+=std[i]*(int(point_num[i])-2)/point_num_sum

print("最大偏移距离"+str(max_det))
# print(min_det)
# print(np.mean(avg_res))
print("平均偏移距离"+str(avg_det))
# print(np.mean(median))
print("平均偏移距离中位数"+str(avg_mediem))
# print(np.mean(std))
print("平均偏移距离标准差"+str(avg_std))
#endregion

# x_train_length=get_length(x_train)
#
# print(x_train_length)
# print(np.max(x_train_length))
# print(np.min(x_train_length))