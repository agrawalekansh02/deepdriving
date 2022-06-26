import torch, glob, cv2
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class CamVidDataset(Dataset):
    def __init__(self, shape, IMAGE_PATH, MASK_PATH, label_dict):
        self.images = glob.glob(IMAGE_PATH)
        self.labels = glob.glob(MASK_PATH)
        self.label_dict = label_dict
        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(shape),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(shape)
        ])
        self.images.sort()
        self.labels.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.img_transform(img)
        mask = cv2.imread(self.labels[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.mask_transform(mask)
        mask = np.array(mask)
        mask = self.adjust_mask(mask, self.label_dict)
        mask = torch.tensor(mask)
        mask = torch.squeeze(mask, dim=0)
        return img, mask

    def adjust_mask(self, mask, label_dict):
        segmentation_map_list = []
        for x,color in enumerate(label_dict.values()):
            segmentation_map = (mask==color).all(axis=-1)
            segmentation_map=(segmentation_map*1)
            segmentation_map*=x
            segmentation_map_list.append(segmentation_map)
            
        return np.amax(np.stack(segmentation_map_list,axis=-1),axis=-1)

    def convert_n_channels_2_rgb(self, image, label_dict):
        image = np.amax(image,axis=-1)
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        
        for l in label_dict.keys():
            idx = image==l
            r[idx] = label_dict[l][0]
            g[idx] = label_dict[l][1]
            b[idx] = label_dict[l][2]
        return np.stack([r,g,b],axis=-1)