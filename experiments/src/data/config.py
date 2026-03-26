from utils.DotDict import DotDict
from config import config as root_config

import os

config = DotDict()
config.update(root_config)

config.update(
    DotDict(
        csv_root_path = os.path.join(
            config.root_path,
            'data'
        ),
        # csv_root_path = "/home/miruna/Skin-FSL/repo/Experiments/data/datasets/ISIC18-T3/ds_phase_1/",
        data_root_path = os.path.join(
            config.root_path, 
            'data', 
            'dataset'
        ),
    )
)

config.update(
    DotDict(
        isic18_t3_root_path = os.path.join(
            config.data_root_path, 
            'ISIC18-T3'
        ),
        derm7pt_root_path = '/home/miruna/Skin-FSL/Derm7pt/release_v0',
		ph2_root_path = os.path.join(
            config.data_root_path, 
            'PH2_Dataset'
        )
    )
)

config.update(
    DotDict(               
        isic18_t3_train_path = os.path.join(
            config.isic18_t3_root_path, 
            'train'
        ),
        isic18_t3_val_path = os.path.join(
            config.isic18_t3_root_path, 
            'val'
        ),
        isic18_t3_test_path = os.path.join(
            config.isic18_t3_root_path, 
            'test'
        )
    )
)

config.update(
    DotDict(
        test_classes = [
            'AKIEC',
            'VASC',
            'DF'
        ],
        train_classes = [
            'MEL',
            'NV',
            'BCC',
            'BKL'
        ],
        derm7pt_test_classes = [
            'BCC',
            'SK',
            'MISC'
        ],
        derm7pt_train_classes = [
            'NEV',
            'MEL'
        ]
    )
)
# 在 config.py 中更新或添加以下内容
# --- ISIC 2019 路径配置 ---
config.update(
    DotDict(
        isic19_root_path = config.data_root_path,
        # 对应你在 Google Drive 上的存放路径
        #isic19_root_path = '/content/drive/MyDrive/datasets/isic2019/ISIC_2019_Training_Input',
        # 注意：CSV 文件的读取逻辑通常依赖于这个根路径下的 data 文件夹
        isic19_csv_path =os.path.join(config.csv_root_path,'ISIC_2019_Training_GroundTruth')
    )
)

config.update(
    DotDict(               
        isic19_train_path = os.path.join(config.isic19_root_path, 'train'),
        isic19_val_path = os.path.join(config.isic19_root_path, 'val'),
        isic19_test_path = os.path.join(config.isic19_root_path, 'test')
    )
)

# --- ISIC 2019 类别划分 (Few-shot 逻辑) ---
config.update(
    DotDict(
        # 训练类 (Base Classes): 模型用来学习特征的类别
        isic19_train_classes = [
            'MEL',   # 黑色素瘤
            'NV',    # 黑色素细胞痣
            'BCC',   # 基底细胞癌
            'BKL'    # 良性角化病
            'AK',    # 光化性角化病
        ],
        # 测试类 (Novel Classes): 模型训练时完全不可见，用于评估 Few-shot 性能
        # 包含 2019 新增的 SCC 以及其他类别
        isic19_test_classes = [
            
            'DF',    # 皮肤纤维瘤
            'VASC',  # 血管病变
            'SCC'    # 鳞状细胞癌 (2019新增)
           
        ]
    )
)