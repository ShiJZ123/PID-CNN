import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
import random

class PositionDataset(Dataset):
    def __init__(self,dataset_path):
        # 获取对应图片位置
        self.position = pd.read_csv(dataset_path+"/position.csv",index_col=[0])
        self.position = self.position.values
        self.position = torch.tensor(self.position ).float()
        # 归一化坐标
        # 理论值
        self.mean = 0.0
        self.std = 28.86751345948128822

        # 获取两个视角图片地址
        self.add= []
        for fname in (os.listdir(os.path.join(dataset_path,'L'))):
            self.add.append(os.path.join(dataset_path,'L',fname))

        # 图像转化操作
        self.resize = transforms.Resize((256,256))
        self.totensor = transforms.ToTensor()

        # 背景信息
        self.back_L = cv2.imread(dataset_path+'\Background_L.png')
        self.back_R = cv2.imread(dataset_path+'\Background_R.png')

        self.view = []
        for i in range(len(self.add)):
            buffer_L1 = self.add[i]
            buffer_R1 = buffer_L1.replace('\\L\\L', '\\R\\R')
            # 读取图像
            buffer_L1 = cv2.imread(buffer_L1)
            buffer_R1 = cv2.imread(buffer_R1)
            # 减去背景信息
            buffer_L1 = buffer_L1 - self.back_L
            buffer_R1 = buffer_R1 - self.back_R
            # 转化为tensor格式
            buffer_L1 = self.totensor(buffer_L1[:, :, 0])  # torch.tensor(buffer_R_chw,dtype=torch.float32) #torch.from_numpy(buffer_L_chw)
            buffer_R1 = self.totensor(buffer_R1[:, :, 0])  # torch.tensor(buffer_R_chw,dtype=torch.float32) #torch.from_numpy(buffer_L_chw)
            # Resize
            buffer_L1 = self.resize(buffer_L1)
            buffer_R1 = self.resize(buffer_R1)
            # stack
            buffer = torch.concat((buffer_L1, buffer_R1), 0).reshape(2, 1, 256, 256)
            self.view.append(buffer)
        pass

    def __len__(self):
        return  len(self.add)

    def __getitem__(self,index):
        index2 = random.randrange(0, len(self.add))
        index3 = random.randrange(0, len(self.add))
        buffer1 = self.view[index]
        buffer2 = self.view[index2]
        buffer3 = self.view[index3]
        buffer = torch.concat((buffer1, buffer2, buffer3), 1)

        p1 = self.position[index]
        p2 = self.position[index2]
        p3 = self.position[index3]
        v1 = p2 - p1
        v2 = p3 - p2
        a = v2 - v1
        p_raw = torch.concat((p1, p2, p3, v1, v2, a), 0)
        p_norm = (p_raw - self.mean) / self.std
        # 返回tensor格式的双视角图片，归一化目标坐标，原始坐标
        index_c = torch.tensor((index, index2, index3))
        return index_c, buffer, p_raw, p_norm


if __name__ == '__main__':

    train_data = PositionDataset(dataset_path=r"D:\ProgramData\pydata\motion_data_100\val_data_motion")
    train_dataloader = DataLoader(train_data,batch_size=2,shuffle=True,num_workers=0)
    index,view,pos_raw,pos_norm = train_data[0]

pass