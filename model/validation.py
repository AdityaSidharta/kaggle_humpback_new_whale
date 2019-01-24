import numpy as np
import pandas as pd
import torch
from tqdm import tqdm_notebook as tqdm

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


def new_whale_threshold(low, high, step, model, criterion, loss_fn, metric_fn, val_dataloader):
    row = []
    for threshold in tqdm(np.arange(low, high+1e-8, step), total=len(np.arange(low, high, step))):
        _, metric = validate_model(model, criterion, loss_fn, metric_fn, val_dataloader, threshold=threshold)
        row.append({
            'threshold': threshold,
            'metric': metric
        })
    return pd.DataFrame(row)
