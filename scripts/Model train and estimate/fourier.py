import math
import array
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split


import time
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'


yasuolv=15
# items_nums=int((((200*yasuolv/100)/2)-1)/2)
items_nums=25


 # 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 data 文件夹路径（向上两级）
data_dir = os.path.join(current_dir, '..', '..', 'data')

# 转为绝对路径，避免路径问题
# data_dir = os.path.join(data_dir,'')

det_path = os.path.join(data_dir,'det_topo_part')
point_path = os.path.join(data_dir,'point_topo_part')

sample_path = os.path.join(data_dir,'det_topo_part')
output_path2 = os.path.join(data_dir,'fourier')
# sample_path = "C:\\Users\\ZS-1PC\\Desktop\\数据与代码\\海洋面All Step Data\\岛群\\det_topo_part"
# output_path2="C:\\Users\\ZS-1PC\\Desktop\\数据与代码\\海洋面All Step Data\\岛群\\fourier"


class Fourier:


    def __init__(self,sample,point_num):
        x, y, xmin, xmax, ymin, ymax, length, sum_length, area, triangle=draw_det(sample,point_num)
        self.ori_x = x
        self.ori_y = y
        self.x,self.y,self.length=self.mirror(self.ori_x,self.ori_y,length)
        self.sum_length = 2 * sum_length
        self.s=[]
        self.s.append(0.0)
        sum=0.0
        for i in range(len(self.length)-1):
            sum+=self.length[i]
            self.s.append(sum)


        self.s.append(self.sum_length)



    def mirror(self,x,y,length):
        ori_x=x
        ori_y=y
        ori_length=length
        if len(x)<=2:
            return
        else:
            center_x=(x[0]+x[len(x)-1])/2
            center_y =(y[0]+y[len(x)-1])/2
            for i in range(len(x)-1):
                ori_x.append(2*center_x-x[i+1])
                ori_y.append(2 * center_y - y[i + 1])
                ori_length.append(length[i])
            return ori_x,ori_y,ori_length


    def ratio_A0_x(self):
        sum=0.0
        for i in range(len(self.length)):
            d_chu_s=(self.x[i+1]-self.x[i])/(self.s[i+1]-self.s[i])

            part1=0.5*d_chu_s*(self.s[i+1]*self.s[i+1]-self.s[i]*self.s[i])
            part2 =(self.x[i]-d_chu_s*self.s[i])*(self.s[i+1]-self.s[i])
            res=part1+part2
            sum+=res

        a=2/self.sum_length*sum
        return a

    def ratio_A0_y(self):
        sum = 0.0
        for i in range(len(self.length)):
            d_chu_s = (self.y[i + 1] - self.y[i]) / (self.s[i + 1] - self.s[i])

            part1 = 0.5 * d_chu_s * (self.s[i + 1] * self.s[i + 1] - self.s[i] * self.s[i])
            part2 = (self.y[i] - d_chu_s * self.s[i] )* (self.s[i + 1] - self.s[i])
            res = part1 + part2
            sum += res

        a=2/self.sum_length*sum
        return a

    def ratio_An_x(self,n):
        sum=0.0
        for i in range(len(self.length)):
            d_chu_s=(self.x[i+1]-self.x[i])/(self.s[i+1]-self.s[i])
            l_chu_2npi=self.sum_length/(2*n*math.pi)
            part1=(self.x[i]-d_chu_s*self.s[i])*l_chu_2npi*(math.sin(self.s[i+1]/l_chu_2npi)-math.sin(self.s[i]/l_chu_2npi))
            part2 =d_chu_s*l_chu_2npi*l_chu_2npi*(self.s[i+1]/l_chu_2npi*math.sin(self.s[i+1]/l_chu_2npi)-self.s[i]/l_chu_2npi*math.sin(self.s[i]/l_chu_2npi)+math.cos(self.s[i+1]/l_chu_2npi)-math.cos(self.s[i]/l_chu_2npi))
            res=part1+part2
            sum+=res

        a=2/self.sum_length*sum
        return a


    def ratio_Bn_x(self,n):
        sum = 0.0
        for i in range(len(self.length)):
            d_chu_s = (self.x[i + 1] - self.x[i]) / (self.s[i + 1] - self.s[i])
            l_chu_2npi = self.sum_length/ (2 * n * math.pi)
            part1 = (self.x[i] - d_chu_s * self.s[i])*l_chu_2npi * (-1)*(
                        math.cos(self.s[i + 1] / l_chu_2npi) - math.cos(self.s[i] / l_chu_2npi))
            part2 = d_chu_s *l_chu_2npi* l_chu_2npi * (math.sin(
                    self.s[i + 1] / l_chu_2npi) - math.sin(self.s[i] / l_chu_2npi)-(
                        self.s[i + 1] / l_chu_2npi * math.cos(self.s[i + 1] / l_chu_2npi) - self.s[
                    i] / l_chu_2npi * math.cos(self.s[i] / l_chu_2npi)))
            res = part1 + part2
            sum += res

        b=2/self.sum_length*sum
        return b


    def ratio_An_y(self,n):
        sum=0.0
        for i in range(len(self.length)):
            d_chu_s=(self.y[i+1]-self.y[i])/(self.s[i+1]-self.s[i])
            l_chu_2npi=self.sum_length/(2*n*math.pi)
            part1=(self.y[i]-d_chu_s*self.s[i])*l_chu_2npi*(math.sin(self.s[i+1]/l_chu_2npi)-math.sin(self.s[i]/l_chu_2npi))
            part2 =d_chu_s*l_chu_2npi*l_chu_2npi*(self.s[i+1]/l_chu_2npi*math.sin(self.s[i+1]/l_chu_2npi)-self.s[i]/l_chu_2npi*math.sin(self.s[i]/l_chu_2npi)+math.cos(self.s[i+1]/l_chu_2npi)-math.cos(self.s[i]/l_chu_2npi))
            res=part1+part2
            sum+=res

        a=2/self.sum_length*sum
        return a


    def ratio_Bn_y(self,n):
        sum = 0.0
        for i in range(len(self.length)):
            d_chu_s = (self.y[i + 1] - self.y[i]) / (self.s[i + 1] - self.s[i])
            l_chu_2npi = self.sum_length/ (2 * n * math.pi)
            part1 = (self.y[i] - d_chu_s * self.s[i])*l_chu_2npi * (-1)*(
                        math.cos(self.s[i + 1] / l_chu_2npi) - math.cos(self.s[i] / l_chu_2npi))
            part2 = d_chu_s *l_chu_2npi* l_chu_2npi * (math.sin(
                    self.s[i + 1] / l_chu_2npi) - math.sin(self.s[i] / l_chu_2npi)-(
                        self.s[i + 1] / l_chu_2npi * math.cos(self.s[i + 1] / l_chu_2npi) - self.s[
                    i] / l_chu_2npi * math.cos(self.s[i] / l_chu_2npi) ))
            res = part1 + part2
            sum += res

        b=2/self.sum_length*sum
        return b

    def get_F_line(self,item_num,resp_num):
        x_res=[]
        y_res= []
        gap=self.sum_length/(resp_num-1)
        A0x=self.ratio_A0_x()
        A0y=self.ratio_A0_y()

        An_x=[]
        Bn_x = []
        An_y = []
        Bn_y = []
        for j in range(item_num):
            n = j + 1
            An_x.append(self.ratio_An_x(n))
            Bn_x.append(self.ratio_Bn_x(n))
            An_y.append(self.ratio_An_y(n))
            Bn_y.append(self.ratio_Bn_y(n))



        # print(self.sum_length)
        for i in range(resp_num):
            s=i*gap
            # print(s)
            x=0.5*A0x
            y =0.5*A0y
            for j in range(item_num):
                n=j+1
                x=x+An_x[j]*math.cos(2*n*math.pi*s/self.sum_length)+Bn_x[j]*math.sin(2*n*math.pi*s/self.sum_length)
                y=y+An_y[j]*math.cos(2*n*math.pi*s/self.sum_length)+Bn_y[j]*math.sin(2*n*math.pi*s/self.sum_length)
            x_res.append(x)
            y_res.append(y)
        return x_res,y_res



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


def get_sample(folder_path):
    file_list = os.listdir(folder_path)
    sample_num = len(file_list)
    all_data = []  # 存储所有文件的数据
    # 遍历文件夹中的每个文件
    for i in range(int(sample_num)):
        file_name = "diff_" + str(i + 1) + "_resampled_coordinates.txt"
        # file_name=str(i+1)+"_resampled_coordinates.txt"
        file_path = folder_path + "\\" + file_name
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
        # 文件有100个数据点
        if len(data) == 200:
            reshaped_data = np.array(data).reshape(200, 1)  # 将其转换为400x1
            all_data.append(reshaped_data)
        else:
            print(f"文件 {file_name} 的数据长度不是400，跳过该文件。")
    # 将所有文件的数据堆叠在一起，形成一个三维矩阵 (文件数, 400, 1)
    all_data_array = np.stack(all_data)
    return all_data_array

def draw_det(data,point_num):
    det_x = [float(data[2 * i]) for i in range(int(point_num)-1)]
    det_y = [float(data[2 * i + 1]) for i in range(int(point_num)-1)]

    point_x = 0.0
    point_y = 0.0
    x = []
    y = []
    triangle = []
    length=[]
    sum_length = 0.0
    area=0.0

    x.append(point_x)
    y.append(point_y)

    for i in range(int(point_num)-1):
        now_length=math.sqrt(math.pow(det_x[i],2)+math.pow(det_y[i],2))
        point_x += det_x[i]
        point_y += det_y[i]
        x.append(point_x)
        y.append(point_y)
        length.append(now_length)
        sum_length+=now_length
        area += (det_x[i]*det_y[i])/2
        # triangle.append(math.atan(det_y[i]/det_x[i]))

    # x.append(0.0)
    # y.append(0.0)

    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)

    return x, y, xmin, xmax, ymin, ymax,length,sum_length,area,triangle



# x_train=get_sample(sample_path)
# point_num=get_point_num(sample_path)
#
# startTime=time.time()
# avg=0.0
# for i in range(x_train.shape[0]):
#
#     m_x_train=x_train[i]
#     m_point_num=point_num[i]
#     f=Fourier(m_x_train,m_point_num)
#     x2,y2=f.get_F_line(50,int(m_point_num)*2-1)
#
#
#
#     x, y, xmin, xmax, ymin, ymax,length,sum_length,area,triangle = draw_det(m_x_train,m_point_num)
#
#     x=np.mat(x)
#     y = np.mat(y)
#     x2 = np.mat(x2[:int(m_point_num)])
#     y2 = np.mat(y2[:int(m_point_num)])
#
#     detx=x2-x
#     dety=y2-y
#     detpoint = np.sqrt(np.multiply(detx, detx) + np.multiply(dety, dety))
#     avg_det=np.mean(detpoint)
#     avg+=avg_det
#     # print(avg_det)
# print(avg/x_train.shape[0])
# endtime = time.time()
# # 3-获取时间间隔
# diffrentTime = endtime - startTime
# print(diffrentTime)

def get_start_point(path):
    file_list=os.listdir(path)
    sample_num=len(file_list)/2
    txt0_path = path + "\\0_point.txt"
    with open(txt0_path, "r") as f:  # 打开文件
        ori_data = f.read()  # 读取文件
        ori_data = ori_data.strip("\n").split("\n")
        data =[]
        data.append(ori_data[0])
    sample=np.array(data)
    for i in range(int(sample_num)):
        if(i!=0):
            txt_path=path+"\\"+str(i)+"_point.txt"
            with open(txt_path, "r") as f:  # 打开文件
                ori_data = f.read()  # 读取文件
                ori_data = ori_data.strip("\n").split("\n")
                data = []
                data.append(ori_data[0])
            new_sample = np.array(data)
            sample=np.row_stack((sample,new_sample))
    return sample

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

def pingcha(X_data,X_text,point_num):
    res=[]
    # 编码解码后
    data = X_data
    input_size=len(data)
    pnum=int(point_num)
    det_x = [float(data[2 * i]) for i in range(int(input_size/2))]
    det_y = [float(data[2 * i + 1]) for i in range(int(input_size/2))]
    point_x = 0.0
    point_y = 0.0
    x = []
    y = []
    x.append(point_x)
    y.append(point_y)

    for j in range(int(input_size/2)):
        point_x += det_x[j]
        point_y += det_y[j]
        x.append(point_x)
        y.append(point_y)
    # 解码编码前
    data2 = X_text
    det_x2 = [float(data2[2 * i]) for i in range(int(input_size/2))]
    det_y2 = [float(data2[2 * i + 1]) for i in range(int(input_size/2))]
    point_x2 = 0.0
    point_y2 = 0.0
    x2 = []
    y2 = []
    x2.append(point_x2)
    y2.append(point_y2)

    for j in range(int(input_size/2)):
        point_x2 += det_x2[j]
        point_y2 += det_y2[j]
        x2.append(point_x2)
        y2.append(point_y2)

#计算插值

    # sum_det_x = x[0] - x[100]-(float(str(start_point[i][0]).split(",")[0]) - float(str(last_point[i][0]).split(",")[0]))/sample_length[i][0]
    # sum_det_y = y[0] - y[100]-(float(str(start_point[i][0]).split(",")[1]) - float(str(last_point[i][0]).split(",")[1]))/sample_length[i][0]
    sum_det_x = x[0] - x[pnum-1] - (x2[0] - x2[pnum-1])
    sum_det_y = y[0] - y[pnum-1] - (y2[0] - y2[pnum-1])
#平均分配
    if  (pnum-1==0):
        if (res == []):
            res = data

        else:
            res = np.row_stack((res, data))
    else:

        x_revise=sum_det_x/(pnum-1)
        y_revise=sum_det_y/(pnum-1)
    #修改
        det_x_res = []
        det_y_res = []
        for i in range(int(input_size/2)):
            det_x_res.append(det_x[i]+x_revise)
            det_y_res.append(det_y[i] + y_revise)

        res_data=[]
        for j in range(int(input_size/2)):
            res_data.append(det_x_res[j])
            res_data.append(det_y_res[j])


        if (res==[]):
            res=res_data

        else:
            res = np.row_stack((res, res_data))

    return res

def arcgis(data,start_point,pointnum):
    det_x = [float(data[2 * i]) for i in range(int(pointnum)-1)]
    det_y = [float(data[2 * i + 1]) for i in range(int(pointnum)-1)]

    start_point_x = str(start_point).split(",")[0]
    start_point_y = str(start_point).split(",")[1]
    point_x = float(start_point_x)
    point_y = float(start_point_y)
    x = []
    y = []
    x.append(point_x)
    y.append(point_y)

    for i in range(int(pointnum)-1):
        point_x += det_x[i]
        point_y += det_y[i]
        x.append(point_x)
        y.append(point_y)
    return x,y

def get_det(x,y,m_point_num):
    res=[]
    for i in range(m_point_num-1):
        res.append(x[i+1]-x[i])
        res.append(y[i + 1] - y[i])
    return res

x_train=get_sample(det_path)
point_num=get_point_num(point_path)
start_point=get_start_point(point_path)

allx=[]
ally=[]

startTime=time.time()
avg=0.0
for i in range(x_train.shape[0]):
    m_start_point=start_point[i]
    m_x_train=x_train[i]
    m_point_num=point_num[i]
    f=Fourier(m_x_train,m_point_num)
    x2,y2=f.get_F_line(items_nums,int(m_point_num)*2-1)

    allx.append(x2)
    ally.append(y2)

endtime = time.time()
# 3-获取时间间隔
diffrentTime = endtime - startTime
print(diffrentTime)

for i in range(x_train.shape[0]):
    x2=allx[i]
    y2 = ally[i]
    m_start_point = start_point[i]
    m_x_train = x_train[i]
    m_point_num = point_num[i]
    x2 = x2[:int(m_point_num)]
    y2 = y2[:int(m_point_num)]


    res_sam=get_det(x2,y2,int(m_point_num))
    x, y, xmin, xmax, ymin, ymax,length,sum_length,area,triangle = draw_det(m_x_train,m_point_num)
    res=pingcha(res_sam,m_x_train,m_point_num)
    x3, y3, xmin3, xmax3, ymin3, ymax3, length3, sum_length3, area3, triangle3 = draw_det(res, m_point_num)


    x = np.mat(x)
    y = np.mat(y)
    x2 = np.mat(x2)
    y2= np.mat(y2)

    detx = x2 - x
    dety = y2 - y

    detpoint = np.sqrt(np.multiply(detx, detx) + np.multiply(dety, dety))
    real_detpoint = detpoint[:, 1:-1]

    avg_det=np.mean(real_detpoint)
    avg+=avg_det


    x3, y3 = arcgis(res, m_start_point,m_point_num)

    with open(output_path2 + "\\" + str(i) + "_point.txt", "w") as f:  # 打开文件
        for j in range(len(x3)):
            f.write(str(x3[j]) + "," + str(y3[j]) + "\n")
    with open(output_path2 + "\\" + str(i) + "_det.txt", "w") as f:  # 打开文件
        for j in range(len(res)):
            f.write(str(res[j]) + ",")

print(avg/x_train.shape[0])



# for i in range(start_point.shape[0]):









# for i in range(10):
#     j=random.randint(0,x_train.shape[0])
#     m_x_train=x_train[j]
#     m_point_num=point_num[j]
#     f=Fourier(m_x_train,m_point_num)
#     x2,y2=f.get_F_line(100,int(m_point_num)*2-1)
#
#     # plt.plot(x2,y2)
#     # # plt.plot(x,y)
#     # plt.plot(f.x,f.y)
#     #
#     #
#     # print(x2)
#     # print(y2)
#
#     # x, y, xmin, xmax, ymin, ymax,length,sum_length,area,triangle = draw_det(m_x_train,m_point_num)
#
#
#
#
#     # plt.legend(['old','50'], loc='upper left')
#     # # plt.tight_layout()
#     # plt.show()
#
# endtime = time.time()
# # 3-获取时间间隔
# diffrentTime = endtime - startTime
# print(diffrentTime)













# x_train=x_train[3]
# point_num=point_num[3]
# f=Fourier(x_train,point_num)
#
# x2,y2=f.get_F_line(100,101)
#
#
# # res50x=res[:-1:2]
# # res50y=res[1::2]
#
#
# x, y, xmin, xmax, ymin, ymax,length,sum_length,area,triangle = draw_det(x_train,point_num)
#
#
# plt.plot(x2,y2)
# # plt.plot(x,y)
# plt.plot(f.x,f.y)
#
#
# print(x2)
# print(y2)
#
#
#
#
#
#
# plt.legend(['old','50'], loc='upper left')
# # plt.tight_layout()
# plt.show()


