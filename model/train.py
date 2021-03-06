import datetime
import os

import torch
from tqdm import tqdm_notebook as tqdm

from model.validation import validate_model
from utils.checkpoint import save_checkpoint, save_metadata
from utils.common import get_batch_info
from utils.envs import model_cp_path


def train_step(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def fit_model(
    model,
    n_epoch,
    dev_dataloader,
    optimizer,
    criterion,
    loss_fn,
    metric_fn,
    val_dataloader=None,
    checkpoint=False,
    model_filename="checkpoint",
    **kwargs
):
    cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    if not os.path.exists(os.path.join(model_cp_path, cur_time)):
        os.mkdir(os.path.join(model_cp_path, cur_time))
    save_metadata(cur_time, model, n_epoch, dev_dataloader, optimizer, criterion, val_dataloader)
    n_dev_obs, dev_batch_size, dev_batch_per_epoch = get_batch_info(dev_dataloader)
    for idx_epoch in tqdm(range(n_epoch), total=n_epoch):
        t = tqdm(enumerate(dev_dataloader), total=dev_batch_per_epoch)
        for idx_batch, data in t:
            model = model.train()
            loss = loss_fn(model, criterion, data)
            train_step(optimizer, loss)
            with torch.no_grad():
                model = model.eval()
                metric = metric_fn(model, data)
            t.set_postfix({"loss": loss.item(), "metric": metric.item()})
        if val_dataloader is not None:
            val_loss, val_metric = validate_model(
                model, criterion, loss_fn, metric_fn, val_dataloader
            )
            print(" val_loss : {}, val_metric : {}".format(val_loss, val_metric))
        if checkpoint:
            filename = "{}_{}".format(model_filename, idx_epoch)
            save_checkpoint(model, optimizer, cur_time, filename)
    return model


def fit_model_full(
    model,
    n_epoch,
    dev_dataloader,
    optimizer,
    criterion,
    loss_fn,
    metric_fn,
    callbacks=[],
    val_dataloader=None,
):
    n_dev_obs, dev_batch_size, dev_batch_per_epoch = get_batch_info(dev_dataloader)
    [cb.on_train_begin(model, optimizer) for cb in callbacks]
    for idx_epoch in tqdm(range(n_epoch), total=n_epoch):
        [cb.on_epoch_begin(idx_epoch, model, optimizer) for cb in callbacks]
        t = tqdm(enumerate(dev_dataloader), total=dev_batch_per_epoch)
        for idx_batch, data in t:
            [cb.on_batch_begin(idx_batch, model, optimizer) for cb in callbacks]
            model = model.train()
            loss = loss_fn(model, criterion, data)
            train_step(optimizer, loss)
            with torch.no_grad():
                model = model.eval()
                metric = metric_fn(model, data)
                t.set_postfix({"loss": loss.item(), "metric": metric.item()})
                [
                    cb.on_batch_end(
                        idx_batch, model, optimizer, loss.item(), metric.item()
                    )
                    for cb in callbacks
                ]
        if val_dataloader is not None:
            val_loss, val_metric = validate_model(
                model, criterion, loss_fn, metric_fn, val_dataloader
            )
            print(" val_loss : {}, val_metric : {}".format(val_loss, val_metric))
            [
                cb.on_epoch_end(idx_epoch, model, optimizer, val_loss, val_metric)
                for cb in callbacks
            ]
        else:
            [cb.on_epoch_end(idx_epoch, model, optimizer) for cb in callbacks]
    [cb.on_train_end(model, optimizer) for cb in callbacks]
    return model
