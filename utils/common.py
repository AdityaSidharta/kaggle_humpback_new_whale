import math
import torch
from sklearn.preprocessing import OneHotEncoder


def remove_new_whale(df):
    return df[df.Id != 'new_whale'].copy()


def create_label(df):
    label = df.Id.values.reshape(-1, 1)
    ohe_model = OneHotEncoder().fit(label)
    ohe_label = ohe_model.transform(label)
    image_label = df.Image.values
    return ohe_model, image_label.tolist(), ohe_label.toarray()


def get_label(df, ohe_model):
    label = df.Id.values.reshape(-1, 1)
    ohe_label = ohe_model.transform(label)
    image_label = df.Image.values
    return image_label.tolist(), ohe_label.toarray()


def get_batch_info(dataloader):
    n_obs = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    n_batch_per_epoch = math.ceil(n_obs / float(batch_size))
    return n_obs, batch_size, n_batch_per_epoch


def img2tensor(img_array, device):
    img_array = img_array.transpose((2, 0, 1))
    return torch.from_numpy(img_array).float().to(device)

def create_dev_val(input_list):
    None