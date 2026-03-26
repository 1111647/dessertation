import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import io, transforms as T
from .config import config

class ISIC19_Dataset(Dataset):
    # ISIC 2019 官方 8 个主要类别索引映射
    class_id_map = dict(MEL=0, NV=1, BCC=2, AK=3, BKL=4, DF=5, VASC=6, SCC=7)

    def __init__(self, mode='train', transform=None, use_cache=True, cache_size=(128, 128)):
        """
        Args:
            mode: 'train', 'val', 或 'test'
            transform: 实时数据增强变换 (如随机翻转、色彩抖动)
            use_cache: 是否开启 RAM 缓存
            cache_size: 缓存到内存前的缩放尺寸 (非常重要，防止 RAM 崩溃)
        """
        super(ISIC19_Dataset, self).__init__()
        
        # 1. 根据模式匹配 CSV 和文件夹
        if mode == 'train' or mode == 'ISIC_2019_Training_GroundTruth':
            csv_filename = "ISIC_2019_Training_GroundTruth.csv"
            self.sub_folder = "train"
        else:
            csv_filename = f"ISIC19_{mode}.csv"
            self.sub_folder = mode 
            
        self.csv_path = os.path.join(config.csv_root_path, csv_filename)
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"找不到 CSV 文件: {self.csv_path}")
            
        self.csv_df = pd.read_csv(self.csv_path)
        self.img_concrete_path = os.path.join(config.isic19_root_path, self.sub_folder)
        
        # 2. 预生成标签列表 (供采样器使用)
        self.labels = []
        for idx in range(len(self.csv_df)):
            label_row = self.csv_df.iloc[idx, 1:]
            active_labels = label_row[label_row == 1.0].index
            if len(active_labels) > 0:
                self.labels.append(self.class_id_map[active_labels[0]])
            else:
                self.labels.append(0)

        # 3. 缓存配置
        self.use_cache = use_cache
        self.cache = {}
        self.transform = transform
        
        # 定义缓存前的基础处理：Resize + Float化
        # 只有缩小后的图才进缓存，保证 40GB 够用
        self.cache_preprocess = T.Compose([
            T.Resize(cache_size),
        ])
        
        if self.use_cache:
            print(f"[{mode}] RAM 缓存已启动。图片将缩放至 {cache_size} 后存入内存。")

    @property
    def num_classes(self):
        return len(self.class_id_map)

    def _load_and_cache_image(self, idx):
        """磁盘读取 -> 基础预处理 -> 返回张量"""
        img_name = self.csv_df.iloc[idx, 0]
        img_path = os.path.join(self.img_concrete_path, f"{img_name}.jpg")
        
        # 兼容性处理：防止后缀大小写
        if not os.path.exists(img_path):
            img_path = img_path.replace(".jpg", ".JPG")

        try:
            # 读取原始 uint8 数据 (省内存)
            img_data = io.read_image(img_path)
            
            # 通道转换 (RGBA/Gray -> RGB)
            if img_data.shape[0] == 4:
                img_data = img_data[:3, :, :]
            elif img_data.shape[0] == 1:
                img_data = img_data.repeat(3, 1, 1)

            # 缩放并转为 float32
            img_data = self.cache_preprocess(img_data).float()
            
            # 归一化
            if img_data.max() > 1.0:
                img_data = img_data / 255.0
                
            return img_data
        except Exception as e:
            print(f"警告：读取 {img_path} 失败，返回空张量。错误: {e}")
            return torch.zeros((3, 128, 128))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 获取图像数据 (从缓存或磁盘)
        if self.use_cache and idx in self.cache:
            img_data = self.cache[idx]
        else:
            img_data = self._load_and_cache_image(idx)
            if self.use_cache:
                self.cache[idx] = img_data

        # 获取标签
        label = self.labels[idx]

        # 应用实时变换 (如 RandomFlip, ColorJitter)
        if self.transform:
            img_data = self.transform(img_data)

        return img_data, label

    def __len__(self):
        return len(self.csv_df)