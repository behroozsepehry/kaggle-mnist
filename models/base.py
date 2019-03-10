import numpy as np
import time

import torch
from torch import nn


class ModelBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ModelBase, self).__init__()
        self.save_path = kwargs.get('save_path')
        self.load_path = kwargs.get('load_path')
        self.name = kwargs.get('name')
        self.train_args = kwargs.get('train_args', {})
        self.evaluate_args = kwargs.get('evaluate_args', {})

    def load(self, path=None, **kwargs):
        if not path:
            path = self.load_path
        if path:
            data = torch.load(path, map_location=kwargs.get('map_location'))
            self.load_state_dict(data['state_dict'])

    def save(self, path=None, **kwargs):
        save_data = kwargs.get('save_data', {})
        assert type(save_data) == dict
        if not path:
            path = self.save_path
        if path:
            torch.save(dict(**save_data, state_dict=self.state_dict()), path)

    def train_epoch(self, epoch, optimizer, trainer_loader, loss_func, device, logger, **kwargs):
        t0 = time.time()

        log_interval = kwargs.get('log_interval', self.train_args.get('log_interval', 1))
        verbose = kwargs.get('verbose', self.train_args.get('verbose', False))

        loss_epoch = 0.
        for i_batch, (x, y) in enumerate(trainer_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = self(x)
            loss = loss_func(y_hat, y)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

            if verbose:
                if i_batch % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
                        epoch, i_batch * len(x), len(trainer_loader.sampler),
                        100. * i_batch / len(trainer_loader),
                        loss / len(x)))

        loss_epoch /= len(trainer_loader.sampler)

        if verbose:
            print('====> Epoch: {} Average loss: {}'.format(
                epoch, loss_epoch))
            print('Time: %.2f s' % (time.time()-t0))

        if logger and logger.flags.get('loss'):
            logger.add_scalar('loss/train', loss_epoch, epoch)

        return dict(loss=loss_epoch)

    def train_model(self, device, dataloaders, optimizer, loss_func, logger, **kwargs):
        t0 = time.time()
        n_epochs = kwargs.get('n_epochs', self.train_args.get('n_epochs'))

        val_loss = self.evaluate_epoch(0, dataloaders.get('val'), loss_func, device, logger, name='val')['loss']

        best_val_loss = val_loss
        validated_train_loss = np.inf
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(epoch, optimizer, dataloaders['train'], loss_func, device, logger)['loss']
            val_loss = self.evaluate_epoch(epoch, dataloaders.get('val'), loss_func, device, logger, name='val')['loss']

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                validated_train_loss = train_loss
                self.save(save_data=dict(epoch=epoch,
                                         train_loss=train_loss, val_loss=val_loss))
        print('Validated train/val loss: %.4f/%.4f' % (validated_train_loss, best_val_loss))
        print('Training finished in %.2f s' % (time.time()-t0))
        return best_val_loss

    def evaluate_epoch(self, epoch, tester_loader, loss_func, device, logger, **kwargs):
        t0 = time.time()
        if not tester_loader:
            return dict(loss=np.inf)

        verbose = kwargs.get('verbose', self.evaluate_args.get('verbose', False))
        name = kwargs.get('name', 'test')

        self.eval()
        loss_epoch = 0.
        with torch.no_grad():
            for i_batch, (x, y) in enumerate(tester_loader):
                x = x.to(device)
                y = y.to(device)
                y_hat = self(x)
                loss = loss_func(y_hat, y)
                loss_epoch += loss.item()

        loss_epoch /= len(tester_loader.sampler)
        if verbose:
            print('====> {} set loss: {}'.format(name, loss_epoch))

        if logger and logger.flags.get('loss'):
            logger.add_scalar('loss/%s' % name, loss_epoch, epoch)

        if verbose:
            print('Time: %.2f s' % (time.time()-t0))

        return dict(loss=loss_epoch)

    def evaluate_model(self, *args, **kwargs):
        return self.evaluate_epoch(*args, **kwargs)
