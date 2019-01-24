import numpy as np
import torch


def mapk(target_indices, prediction_indices, k):
    mapk_array = np.zeros(target_indices.shape[0])
    for i in reversed(range(k)):
        mapk_array = np.where(target_indices == prediction_indices[:, i], 1/(i+1), mapk_array)
    return mapk_array


def loss_fn(model, criterion, data, **kwargs):
    img, target = data
    prediction = model(img)
    loss = criterion(prediction, target)
    return loss


def metric_fn(model, data, **kwargs):
    img, target = data
    prediction = torch.sigmoid(model(img))
    if kwargs:
        prediction[:, 0] = kwargs.get('threshold')
    target_indices = torch.topk(target, 1)[1].cpu().numpy().ravel()
    prediction_indices = torch.topk(prediction, 5)[1].cpu().numpy()
    mapk_array = mapk(target_indices, prediction_indices, 5)
    return np.array([np.mean(mapk_array)])


def pred_fn(model, data, **kwargs):
    img = data
    prediction = torch.sigmoid(model(img))
    if kwargs:
        prediction[:, 0] = kwargs.get('threshold')
    prediction_indices = torch.topk(prediction, 5)[1].cpu().numpy()
    return prediction_indices

