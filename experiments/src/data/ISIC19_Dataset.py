# 建议在 experiments/src/data/ 下创建此文件
from torch.utils.data import Dataset
from torchvision import io 
import os 
import pandas as pd
import torch

from .config import config

class ISIC19_Dataset(Dataset):
    class_id_map = dict(MEL=0, NV=1, BCC=2, AK=3, BKL=4, DF=5, VASC=6, SCC=7)

    def __init__(self, mode='train', transform=None):
        super(ISIC19_Dataset, self).__init__()
        
        # 1. 自动适配文件名逻辑
        if mode == 'train' or mode == 'ISIC_2019_Training_GroundTruth':
            csv_filename = "ISIC_2019_Training_GroundTruth.csv"
            self.sub_folder = "train" # 对应你分好的文件夹
        else:
            csv_filename = f"ISIC19_{mode}.csv"
            self.sub_folder = mode    # val 或 test
            
        self.csv_path = os.path.join(config.csv_root_path, csv_filename)
        self.csv_df = pd.read_csv(self.csv_path)
        
        # 2. 自动适配图片子文件夹
        # 路径应为：data/datasets/train (或 val/test)
        self.img_concrete_path = os.path.join(config.isic19_root_path, self.sub_folder)
        self.transform = transform

  
    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        
        # 读取图片名并拼接路径
        img_name = self.csv_df.iloc[idx, 0]
        img_path = os.path.join(self.img_concrete_path, f"{img_name}.jpg")
        
        # 解析 One-hot 标签：找到值为 1.0 的列名
        label_row = self.csv_df.iloc[idx, 1:]
        class_name = label_row[label_row == 1.0].index[0]
        label = self.class_id_map[class_name]

        img_data = io.read_image(img_path).float()
        if self.transform: img_data = self.transform(img_data)
        return img_data, label

    def __len__(self):
        return len(self.csv_df)