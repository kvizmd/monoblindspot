import os

import torch
from torch import nn


class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None

    def load_weight(self, weight: str):
        device = next(self.parameters()).device
        load_weights(self, weight, device)

    def export_weight(self) -> dict:
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def forward(self, x: torch.Tensor, *args, **kwargs) -> dict:
        features = self.encoder(x)
        outputs = self.decoder(features, *args, **kwargs)
        return outputs


def load_weights(model, filename, device, auto_slice: bool = True):
    if not os.path.isfile(filename):
        raise RuntimeError('Not exist ' + str(filename))
    checkpoint = torch.load(filename, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict_ = checkpoint['state_dict']

    elif 'model' in checkpoint:
        state_dict_ = checkpoint['model']

    else:
        state_dict_ = checkpoint

    state_dict = {}
    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]

    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in model_state_dict.keys():
        if k in state_dict:
            load_shape = state_dict[k].shape
            model_shape = model_state_dict[k].shape
            if load_shape != model_shape:
                if auto_slice and load_shape[1:] == model_shape[1:]:
                    # If the shape is just different the output size (of conv),
                    # slice the weights so that the weight matches the model.
                    print(
                        'The shape {} is automatically changed '.format(k)
                        + 'from {} to {}.'.format(
                            str(load_shape), str(model_shape)))
                    state_dict[k] = state_dict[k][:model_shape[0]]
                else:
                    print('Different shape {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        else:
            print('{} does not have {}.'.format(filename, k))
            state_dict[k] = model_state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    # print('Model Loaded: ', filename)
    return model
