import numpy as np
import pandas as pd
from sklearn import model_selection

import torch
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utilities import general_utilities


def get_instance(folder, instance_type, **kwargs):
    kwargs_instance = kwargs
    instance_filename = kwargs_instance.get('name')
    if not instance_filename:
        instance = None
    else:
        module = general_utilities.import_from_path(folder+instance_filename+'.py')
        instance = getattr(module, instance_type)(**kwargs_instance)
    return instance


def get_model(**kwargs):
    return get_instance('models/', 'Model', **kwargs)


def get_optimizer(parameters, **kwargs):
    optimizer_name = kwargs['name']
    optimizer_constructor = getattr(optim, optimizer_name)
    optimizer = optimizer_constructor(parameters, **kwargs['args'])
    return optimizer


def get_dataloaders(**kwargs):
    path = kwargs.get('path')
    train = pd.read_csv(path+'train.csv')

    y_train = train["label"].values
    x_train = train.drop(labels=["label"], axis=1).values
    x_test = pd.read_csv(path+'test.csv').values

    del train

    x_train = x_train / 255.
    x_test = x_test / 255.
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)

    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=kwargs['ratio']['val'])

    dataloaders = {
        'train':
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train)),
                **kwargs['args']),
        'val':
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.Tensor(x_val), torch.LongTensor(y_val)),
                **kwargs['args']),
        'test':
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.Tensor(x_test)),
                **dict(**kwargs['args'], shuffle=False))
    }
    return dataloaders


def get_loss(**kwargs):
    loss_name = kwargs['name']
    loss_constructor = getattr(optim, loss_name)
    loss_func = loss_constructor(**kwargs['args'])
    return loss_func


def get_device(**kwargs):
    device_name = kwargs['name']
    if torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        device_name_2 = 'cpu'
        device = torch.device(device_name_2)
        if device_name_2 != device_name:
            print('Warning: device \'%s\' not available, using device \'%s\' instead'% (device_name, device_name_2))
    return device


def get_logger(**kwargs):
    if kwargs['name'] == 'tensorboard':
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(**kwargs['args'])
        logger.flags = kwargs.get('flags', {})
    elif not kwargs['name']:
        logger = None
    else:
        raise NotImplementedError
    return logger

