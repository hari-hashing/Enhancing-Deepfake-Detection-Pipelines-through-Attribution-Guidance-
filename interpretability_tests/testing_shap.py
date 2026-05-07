import torch,torchvision
import numpy as np
import shap
import argparse
from models.fake_image_model import build_model
from datasets.fake_image_dataset import FakeImageDataset
from utils.augmentations import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader
 
# normalization constants
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# specifying device 
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

# load the model 
model = build_model(unfreeze_at_epoch = None).to(device)

# model loading config 
@classmethod
class config:
    def __init__(self):
        super().__init__()
        self.batch_size = 512
        self.shuffle = True 
        self.num_workers = 8
        # self.pin_memory = True
        self.persistent_workers = True
        self.prefetch_factor = 4

# load the testing dataset
def load_dataset(data_dir,config):
    ds = FakeImageDataset(data_dir, transform=get_val_transforms())
    loader = DataLoader(ds, batch_size = config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers,
                        persistent_workers = config.persistent_workers, prefetch_factor = config.prefetch_factor)
    return ds,loader


# defining tranform functions and normalizations 
def nchw_to_nhwc(x : torch.Tensor) -> torch.Tensor:
    # [n,c,h,w] -> [n,h,w,c]
    if x.ndim == 4:
        x = x if x.shape[1] == 3 else x.permute(0,2,3,4)
    elif x.ndim == 3:
        x = x if x.shape[0] == 3 else x.permute(1,2,0)
    return x
    
transform = [
    torchvision.transforms.Normalize(mean=mean, std=std),
]

inv_transform = [
    torchvision.transforms.Normalize(
        mean=(-1 * np.array(mean) / np.array(std)).tolist(),
        std=(1 / np.array(std)).tolist(),
    ),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]

transform = torchvision.transforms.Compose(transform)
inv_transform = torchvision.transforms.Compose(inv_transform)

def load_model(checkpoint_path):
    checkpoint_path = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint_path['model_state_dict'])
    model.eval()
    print(f"- Loaded model from {checkpoint_path}")
    return model

