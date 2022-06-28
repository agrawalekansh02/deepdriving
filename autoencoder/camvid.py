import glob, cv2
from torch.utils.data import Dataset
from torchvision import transforms

class CamVidDataset(Dataset):
    def __init__(self, shape, IMAGE_PATH):
        self.images = glob.glob(IMAGE_PATH)
        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(shape),
            transforms.ToTensor()
        ])
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.img_transform(img)
        return img, img