import os
from collections import OrderedDict

import torch

from callbacks.callback import CallBacks
from utils.common import get_batch_info, write_yaml
from utils.envs import model_cp_path, result_metadata_path


def save_checkpoint(model, optimizer, foldername, filename=None):
    model_filename = "{}_model.pth".format(filename)
    optim_filename = "{}_optim.pth".format(filename)
    model_filepath = os.path.join(model_cp_path, foldername, model_filename)
    optim_filepath = os.path.join(model_cp_path, foldername, optim_filename)
    torch.save(model.state_dict(), model_filepath)
    torch.save(optimizer.state_dict(), optim_filepath)


def save_metadata(filename, model, n_epoch, dev_dataloader, optimizer, criterion, val_dataloader, scheduler=None):
    file_path = os.path.join(result_metadata_path, '{}.yaml'.format(filename))
    n_dev, bs_dev, _ = get_batch_info(dev_dataloader)

    metadata = OrderedDict()
    metadata['model'] = {
        'name': model.__class__.__name__
    }
    metadata['n_epoch'] = n_epoch
    metadata['train_dataset'] = {
        'n_obs': n_dev,
        'batch_size': bs_dev
    }
    metadata['optimizer'] = {
        'name': optimizer.__class__.__name__,
        'params': optimizer.defaults
    }
    metadata['criterion'] = {
        'name': criterion.__class__.__name__
    }
    if scheduler:
        metadata['scheduler'] = {
            'name': scheduler.__class__.__name__,
            'params': scheduler.state_dict()
        }
    if val_dataloader:
        n_val, bs_val, _ = get_batch_info(val_dataloader)
        metadata['val_dataset'] = {
            'n_obs': n_val,
            'batch_size': bs_val
        }
    write_yaml(metadata, file_path)



def load_cp_model(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)


def load_cp_optim(optimizer, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    optimizer.load_state_dict(state_dict)


class CheckpointSaver(CallBacks):
    def __init__(self, model_fn, is_epoch_cp=True):
        self.model_fn = model_fn
        self.is_epoch_cp = is_epoch_cp

    def on_epoch_end(self, epoch_idx, model, optimizer, val_loss=None, val_metric=None):
        model_filename = "{}_{}".format(self.model_fn, epoch_idx)
        save_checkpoint(model, optimizer, model_filename)

    def on_train_end(self, model, optimizer):
        model_filename = "{}_final", format(self.model_fn)
        save_checkpoint(model, optimizer, model_filename)
