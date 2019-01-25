import numpy as np
import pandas as pd
import torch
from tqdm import tqdm_notebook as tqdm
from model.functions import mapk
from utils.common import get_batch_info


def validate_model(model, criterion, loss_fn, metric_fn, val_dataloader, **kwargs):
    n_val_obs, val_batch_size, val_batch_per_epoch = get_batch_info(val_dataloader)
    total_loss = np.zeros(val_batch_per_epoch)
    total_metric = np.zeros(val_batch_per_epoch)
    model = model.eval()
    t = tqdm(enumerate(val_dataloader), total=val_batch_per_epoch)
    with torch.no_grad():
        for idx, data in t:
            loss = loss_fn(model, criterion, data, **kwargs)
            metric = metric_fn(model, data, **kwargs)
            total_loss[idx] = loss
            total_metric[idx] = metric
    return total_loss.mean(), total_metric.mean()


def new_whale_threshold(low, high, step, model, predict_proba_fn, val_dataloader):
    model = model.eval()
    row = []
    target_array_list = []
    pred_array_list = []
    n_val_obs, val_batch_size, val_batch_per_epoch = get_batch_info(val_dataloader)
    thresh_range = np.arange(low, high+1e-8, step)

    t = tqdm(enumerate(val_dataloader), total=val_batch_per_epoch)
    with torch.no_grad():
        for idx, data in t:
            target, prediction = predict_proba_fn(model, data)
            target_array_list.append(target)
            pred_array_list.append(prediction)
    target_array = np.vstack(target_array_list)
    prediction_array = np.vstack(pred_array_list)

    for threshold in tqdm(thresh_range, total=len(thresh_range)):
        prediction_array[:, 0] = threshold
        prediction_indices = (-prediction_array).argsort()[:, :5]
        mapk_array = mapk(target_array, prediction_indices, 5)
        mapk_result = mapk_array.mean()
        row.append({
            'threshold': threshold,
            'mapk': mapk_result
        })
    return pd.DataFrame(row)
