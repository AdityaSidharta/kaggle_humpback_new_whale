import math
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import oyaml as yaml

def write_yaml(yaml_file, path):
    with open(path, 'w') as outfile:
        yaml.dump(yaml_file, stream=outfile, default_flow_style=False)

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


def inverse_label(array, ohe_model):
    value = ohe_model.categories_[0][array].tolist()
    return ' '.join(value)


def create_kaggle_submission(test_image_label, result_list, ohe_model):
    row = []
    assert len(test_image_label) == len(result_list)
    for idx in range(len(test_image_label)):
        image = test_image_label[idx]
        id = inverse_label(np.array(result_list[idx]), ohe_model)
        row.append({
            'Image': image,
            'Id': id
        })
    return pd.DataFrame(row)[['Image', 'Id']]

