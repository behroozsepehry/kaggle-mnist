import argparse
import yaml
import itertools
import copy
import numpy as np
from scipy import stats

import torch

from utilities import main_utilities as m_util
from utilities import general_utilities as g_util


def construct_objects(settings):
    seed = settings.get('seed')
    if seed is not None:
        torch.manual_seed(seed)

    device = m_util.get_device(**settings['Device'])
    model = m_util.get_model(**settings['Model'])
    model.load(map_location=device)
    model = model.to(device)
    dataloaders = m_util.get_dataloaders(**settings['Dataloaders'])
    optimizer = m_util.get_optimizer(model.parameters(), **settings['Optimizers'])
    loss_func = m_util.get_loss(**settings['Loss'])
    logger = m_util.get_logger(**settings['Logger'])

    if logger and logger.flags.get('conf'):
        logger.add_text('conf/conf', str(settings))

    return device, model, dataloaders, optimizer, loss_func, logger


def train_seed(settings):
    objects = construct_objects(settings)
    device, model, dataloaders, optimizers, losses, logger, evaluator = objects
    result = model.train_model(device, dataloaders, optimizers, losses, logger)
    if evaluator:
        eval = evaluator()
        print('%s\nEvaluated: %s\n%s' % ('#'*10, eval, '#'*10))
    if logger:
        logger.close()
    return result


def train(settings):
    seeds = settings.get('seeds', [settings.get('seed', 0)])
    results = []
    for s in seeds:
        train_settings = copy.deepcopy(settings)
        train_settings['seed'] = s
        r = train_seed(train_settings)
        results.append(r)
    return results


def hypertune(settings):
    hypertune_settings = settings.get('Hypertune', {})
    hypertune_params_settings = hypertune_settings.get('params', {})
    hypertune_params_settings_list_sizes = g_util.get_list_size_of_dict_of_lists(hypertune_params_settings)
    hypertune_params_settings_list_sizes_lists = [range(s) for s in hypertune_params_settings_list_sizes]
    all_possible_choices = itertools.product(*hypertune_params_settings_list_sizes_lists)

    best_mu = np.inf
    best_std = 0.

    for choice in all_possible_choices:
        train_settings = copy.deepcopy(settings)
        g_util.choose_1_from_dict_of_lists(hypertune_params_settings, choice, train_settings)
        vals = train(train_settings)
        mu = np.mean(vals)
        std = np.std(vals, ddof=1)
        n = len(vals)
        df = n-1
        dist = stats.t(df)
        if mu < best_mu:
            # p_value for hypothesis of whether the best and second best hypeparameter are similar
            # 0 means very dissimilar
            p_val = 2. * (1. - dist.cdf(np.abs((mu - best_mu) * np.sqrt(n) / np.linalg.norm([std, best_std]))))
            best_mu = mu
            best_std = std
            best_choice = copy.deepcopy(choice)

    best_hyperparams = copy.deepcopy(hypertune_params_settings)
    g_util.choose_1_from_dict_of_lists(hypertune_params_settings, best_choice, best_hyperparams)

    return dict(best_hyperparams=best_hyperparams, best_mu=best_mu, best_std=best_std, p_val=p_val)


def main():
    parser = argparse.ArgumentParser(description='MNIST classification')
    parser.add_argument('--conf-path', '-c', type=str, default='confs/mnist.yaml', metavar='N',
                        help='configuration file path')
    args = parser.parse_args()
    with open(args.conf_path, 'rb') as f:
        settings = yaml.load(f)

    if settings['function'] == 'train':
        result = train(settings)
    elif settings['function'] == 'hypertune':
        result = train(settings)
    else:
        raise NotImplementedError
    print('result: %s' % (result))


if __name__ == '__main__':
    main()
