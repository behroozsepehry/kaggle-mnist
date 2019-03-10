import torch
from torch import nn


def apply_func_to_model_data(model, func, dataloader, device):
    result = []
    with torch.no_grad():
        for i_batch, data_tuple in enumerate(dataloader):
            x = data_tuple[0]
            x = x.to(device)
            model_out = model(x)
            result.append(func(model_out))
    return result


def data_parallel_model(model, input, ngpu):
    if 'cuda' in str(input.device) and ngpu > 1:
        output = nn.parallel.data_parallel(model, input, range(ngpu))
    else:
        output = model(input)
    return output
