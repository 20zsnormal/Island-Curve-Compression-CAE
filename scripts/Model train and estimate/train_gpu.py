import math
import torch
from torch import nn
from torch.nn import Sequential, Conv1d, MSELoss, BatchNorm1d, ReLU
import time
from sklearn.model_selection import train_test_split
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random

# ----------------------------------------------------------------------
# Device configuration (CPU / GPU)
# ----------------------------------------------------------------------
device = torch.device("cuda")

# ----------------------------------------------------------------------
# Reproducibility settings
# Fix random seeds for numpy, python, and PyTorch
# ----------------------------------------------------------------------
seed_value = 7
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True


# ----------------------------------------------------------------------
# Convolutional Autoencoder (CAE) for polyline encoding and reconstruction
# # ----------------------------------------------------------------------
# class Encode_Decode(nn.Module):
#     """
#     One-dimensional convolutional autoencoder (CAE).
#
#     The network encodes resampled polyline coordinate differences
#     into a low-dimensional latent representation and reconstructs
#     the original signal through symmetric decoding layers.
#     """
#
#     def __init__(self, encoded_space_dim=60):
#         """
#         Parameters
#         ----------
#         encoded_space_dim : int
#             Dimension of the latent (compressed) feature space.
#         """
#         super(Encode_Decode, self).__init__()
#
#         # ---------------- Encoder ----------------
#         self.Encode = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=5, padding=1),
#             nn.BatchNorm1d(16),
#             nn.ReLU(True),
#
#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=5, padding=1),
#             nn.BatchNorm1d(32),
#             nn.ReLU(True),
#         )
#
#         # Flatten convolutional feature maps
#         self.flatten = nn.Flatten(start_dim=1)
#
#         # Fully connected layer for latent representation
#         self.encoder_lin = nn.Sequential(
#             nn.Linear(32 * 8, encoded_space_dim),
#             nn.ReLU(True),
#         )
#
#         # ---------------- Decoder ----------------
#         self.decoder_lin = nn.Sequential(
#             nn.Linear(encoded_space_dim, 32 * 8),
#             nn.ReLU(True),
#         )
#
#         self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 8))
#
#         self.Decode = nn.Sequential(
#             nn.ConvTranspose1d(in_channels=32, out_channels=16,
#                                kernel_size=7, stride=5, padding=1),
#             nn.BatchNorm1d(16),
#             nn.ReLU(True),
#
#             nn.ConvTranspose1d(in_channels=16, out_channels=1,
#                                kernel_size=7, stride=5, padding=1),
#         )
#
#     def forward(self, x):
#         """
#         Forward pass of the autoencoder.
#
#         Parameters
#         ----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, 1, sequence_length)
#
#         Returns
#         -------
#         torch.Tensor
#             Reconstructed signal with the same shape as input.
#         """
#         x = self.Encode(x)
#         x = self.flatten(x)
#         compress = self.encoder_lin(x)
#         x = self.decoder_lin(compress)
#         x = self.unflatten(x)
#         x = self.Decode(x)
#         return x
class Encode_Decode(nn.Module):

    def __init__(self,encoded_space_dim=60):
        super(Encode_Decode, self).__init__()
        # Encoder
        self.Encode = nn.Sequential(
            nn.Linear(200, 150),
            nn.BatchNorm1d(1),  # 添加批量归一化层
            nn.ReLU(True),
            nn.Linear(150, 150),
            nn.BatchNorm1d(1),  # 添加批量归一化层
            nn.ReLU(True),

        )
        ### Linear p
        self.encoder_lin = nn.Sequential(
            nn.Linear( 150, encoded_space_dim),
            nn.BatchNorm1d(1),  # 添加批量归一化层
            nn.ReLU(True),
        )

        # Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 150),
            nn.ReLU(True),

        )
        self.Decode = nn.Sequential(
            nn.Linear(150, 150),
            nn.BatchNorm1d(1),  # 添加批量归一化层
            nn.ReLU(True),
            nn.Linear(150, 200),
        )
    def forward(self,x):
        x=self.Encode(x)
        compress=self.encoder_lin(x)
        x=self.decoder_lin(compress)
        x=self.Decode(x)
        return x
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(f'Selected device: {device}')

# ----------------------------------------------------------------------
# Load polyline data from TXT files
# Each file stores resampled coordinate differences (dx, dy)
# ----------------------------------------------------------------------
def load_txt_files(folder_path):
    """
    Load resampled polyline data from text files.

    Parameters
    ----------
    folder_path : str
        Directory containing txt files.

    Returns
    -------
    np.ndarray
        Array of shape (num_samples, sequence_length, 1)
    """
    file_list = os.listdir(folder_path)
    sample_num = len(file_list)
    all_data = []

    for i in range(sample_num):
        file_name = "diff_" + str(i + 1) + "_resampled_coordinates.txt"
        file_path = folder_path + "\\" + file_name

        data = []
        with open(file_path, 'r') as file:
            for line in file:
                values = list(map(float, line.strip().split(',')))
                data.extend(values)

        # Each sample contains 100 points × (dx, dy) → 200 values
        if len(data) == 200:
            reshaped_data = np.array(data).reshape(200, 1)
            all_data.append(reshaped_data)
        else:
            print(f"File {file_name} has invalid length and is skipped.")

    return np.stack(all_data)


# ----------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 data 文件夹路径（向上两级）
data_dir = os.path.join(current_dir, '..', '..', 'data')

# 转为绝对路径，避免路径问题
# data_dir = os.path.join(data_dir,'')
folder_path = os.path.join(data_dir,'det_topo')
input_array = load_txt_files(folder_path)

# Transpose to (samples, channels, length)
input_Tarray = input_array.transpose(0, 2, 1)

# Train-test split
X_train, X_test = train_test_split(input_Tarray, test_size=0.3, random_state=3)

# Standardization (sample-wise)
mean_train = X_train.mean(axis=(1, 2), keepdims=True)
std_train = X_train.std(axis=(1, 2), keepdims=True) + 1e-8
X_train = (X_train - mean_train) / std_train

mean_test = X_test.mean(axis=(1, 2), keepdims=True)
std_test = X_test.std(axis=(1, 2), keepdims=True) + 1e-8
X_test = (X_test - mean_test) / std_test

# Convert to torch tensors
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)


# ----------------------------------------------------------------------
# Model, loss function, and optimizer
# ----------------------------------------------------------------------
encode_decode = Encode_Decode()
if torch.cuda.is_available():
    encode_decode = encode_decode.cuda()

loss_conv = MSELoss(reduction='mean').cuda()
optimizer = torch.optim.Adam(
    encode_decode.parameters(),
    lr=0.0015,
    weight_decay=1e-5
)

epochs = 200
batch_size = 512


# ----------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------
train_dataset = TensorDataset(X_train, X_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, X_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# ----------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------
diz_loss = {'train_loss': [], 'test_loss': []}
startTime = time.time()

for epoch in range(epochs):
    print(f"----- Epoch {epoch + 1} -----")

    # Training phase
    encode_decode.train()
    train_loss = []

    for input_batch, _ in train_loader:
        input = input_batch.float().cuda()

        output = encode_decode(input)
        loss = loss_conv(output, input)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    avg_train_loss = np.mean(train_loss)
    print(f"Average training loss: {avg_train_loss}")

    # Evaluation phase
    encode_decode.eval()
    test_loss = []

    with torch.no_grad():
        for input_batch, _ in test_loader:
            input = input_batch.float().cuda()
            output = encode_decode(input)
            loss = loss_conv(output, input)
            test_loss.append(loss.item())

    avg_test_loss = np.mean(test_loss)
    print(f"Average test loss: {avg_test_loss}")

    diz_loss['train_loss'].append(avg_train_loss)
    diz_loss['test_loss'].append(avg_test_loss)

    # Save model checkpoint
    # torch.save(
    #     encode_decode,
    #     "C:\\Users\\ZS-1PC\\Desktop\\数据与代码\\海洋面All Step Data\\岛群\\parmspath\\分段75000\\CAE_{}.pth".format(epoch + 1)
    # )

# ----------------------------------------------------------------------
# Training time
# ----------------------------------------------------------------------
endTime = time.time()
print(f"Total training time: {endTime - startTime:.2f} seconds")


# ----------------------------------------------------------------------
# Plot loss curves
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 8))
plt.semilogy(diz_loss['train_loss'], label='Train')
plt.semilogy(diz_loss['test_loss'], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.title('Training and Testing Loss')
plt.show()
