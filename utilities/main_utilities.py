import numpy as np
import pandas as pd
from torchvision.utils import save_image

import torch
from torch import optim
from torch import nn
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


def get_lr_scheduler(optimizer, **kwargs):
    scheduler_name = kwargs['name']
    scheduler_constructor = getattr(optim.lr_scheduler, scheduler_name)
    scheduler = scheduler_constructor(optimizer, **kwargs['args'])
    return scheduler


def get_dataloaders(**kwargs):
    path = kwargs.get('path')

    x_test = pd.read_csv(path+'test.csv').values
    x_test = x_test / 255.
    x_test = x_test.reshape(-1, 1, 28, 28)

    transforms_train = [getattr(transforms, t['name'])(**t['args'])
                        for t in kwargs.get('transforms', {}).get('train', [])] + [transforms.ToTensor()]
    dataset_train = datasets.MNIST(path, train=True, download=True,
                                   transform=transforms.Compose(transforms_train))
    dataset_val = datasets.MNIST(path, train=True, download=True,
                                   transform=transforms.ToTensor())
    dataset_train_size = len(dataset_train)
    idxs = range(dataset_train_size)
    val_ratio = kwargs['ratio'].get('val', 0)
    split_idx = int(np.floor(val_ratio * dataset_train_size))
    val_idxs = idxs[:split_idx]
    train_idxs = idxs[split_idx:]
    val_sampler = SubsetRandomSampler(val_idxs)
    train_sampler = SubsetRandomSampler(train_idxs)

    dataloaders = {
        'train':
            torch.utils.data.DataLoader(
                dataset_train,
                sampler=train_sampler,
                **kwargs['args']),
        'val':
            torch.utils.data.DataLoader(
                dataset_val,
                sampler=val_sampler,
                **kwargs['args']),
        'test':
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.Tensor(x_test)),
                **dict(kwargs['args'], shuffle=False))
    }
    return dataloaders


def get_loss(**kwargs):
    loss_name = kwargs['name']
    loss_constructor = getattr(nn, loss_name)
    loss_func = loss_constructor(**kwargs.get('args', {}))
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


if __name__ == '__main__':
    import argparse
    import yaml
    import os
    os.chdir('..')
    parser = argparse.ArgumentParser(description='MNIST classification')
    parser.add_argument('--conf-path', '-c', type=str, default='confs/mnist.yaml', metavar='N',
                        help='configuration file path')
    args = parser.parse_args()
    with open(args.conf_path, 'rb') as f:
        settings = yaml.load(f)

    dataloaders = get_dataloaders(**settings['Dataloaders'])
    for x,y in dataloaders['train']:
        save_image(x, settings['Dataloaders']['path'] + 'sample_train.png')
        break
    for x,y in dataloaders['val']:
        save_image(x, settings['Dataloaders']['path'] + 'sample_val.png')
        break
    for x, in dataloaders['test']:
        save_image(x, settings['Dataloaders']['path'] + 'sample_test.png')
        break


