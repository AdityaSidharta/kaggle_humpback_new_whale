import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip,\
    RandomVerticalFlip, RandomAffine, Normalize, ToTensor, ToPILImage, Grayscale
from skimage import io
from skimage.color import gray2rgb
import os

train_transform = Compose([
    ToPILImage(),
    Resize((224, 224)),
    Grayscale(3),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomAffine(degrees=30),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = Compose([
    ToPILImage(),
    Resize((224, 224)),
    Grayscale(3),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class TrainDataset(Dataset):
    def __init__(self, image_label, ohe_label, train_path, train_tsfm, device):
        self.image_label = image_label
        self.ohe_label = ohe_label
        self.train_path = train_path
        self.train_tsfm = train_tsfm
        self.device = device

    def __len__(self):
        return len(self.image_label)

    def __getitem__(self, idx):
        img_path = os.path.join(self.train_path, self.image_label[idx])
        img_array = io.imread(img_path)
        if len(img_array.shape) == 2:
            img_array = gray2rgb(img_array)
        assert img_array.shape[2] == 3
        img_tensor = self.train_tsfm(img_array)
        img_tensor = img_tensor.type(torch.float).to(self.device)
        label = self.ohe_label[idx, :]
        label_tensor = torch.Tensor(label)
        label_tensor = label_tensor.type(torch.float).to(self.device)
        return img_tensor, label_tensor


class TestDataset(Dataset):
    def __init__(self, image_label, test_path, test_tsfm, device):
        self.image_label = image_label
        self.test_path = test_path
        self.test_tsfm = test_tsfm
        self.device = device

    def __len__(self):
        return len(self.image_label)

    def __getitem__(self, idx):
        img_path = os.path.join(self.train_path, self.image_label[idx])
        img_array = io.imread(img_path)
        if len(img_array.shape) == 2:
            img_array = gray2rgb(img_array)
        assert img_array.shape[2] == 3
        img_tensor = self.test_tsfm(img_array)
        img_tensor = img_tensor.type(torch.float).to(self.device)
        return img_tensor
