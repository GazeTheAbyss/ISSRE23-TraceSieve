import mltk
import torch
import yaml
from tensorkit import tensor as T

from tracegnn.models.vgae.model import TraceVAE
from tracegnn.models.vgae.train import ExpConfig as TrainConfig
from tracegnn.utils import *
from tracegnn.visualize import *

__all__ = ['load_model']


def load_model(model_path: str,
               id_manager: TraceGraphIDManager,
               strict: bool,
               extra_args
               ) -> TraceVAE:
    # get model file and config file path
    if model_path.endswith('.pt'):
        model_file = model_path
        config_file = model_path.rsplit('/', 2)[-3] + '/config.json'
    else:
        if not model_path.endswith('/'):
            model_path += '/'
        model_file = model_path + 'models/final.pt'
        config_file = model_path + 'config.json'

    # load config
    with as_local_file(config_file) as config_file:
        config_loader = mltk.ConfigLoader(TrainConfig)
        config_loader.load_file(config_file)

    # also patch the config
    if extra_args:
        extra_args_dict = {}
        for arg in extra_args:
            if arg.startswith('--'):
                arg = arg[2:]
                if '=' not in arg:
                    val = True
                else:
                    arg, val = arg.split('=', 1)
                    val = yaml.safe_load(val)
                extra_args_dict[arg] = val
            else:
                raise ValueError(f'Unsupported argument: {arg!r}')
        config_loader.load_object(extra_args_dict)

    # load the model
    if strict:
        discard_undefined = mltk.type_check.DiscardMode.NO
    else:
        discard_undefined = mltk.type_check.DiscardMode.WARN
    config = config_loader.get(discard_undefined=discard_undefined).model
    vae = TraceVAE(config, id_manager.num_operations)
    with as_local_file(model_file) as model_file:
        vae.load_state_dict(torch.load(
            model_file,
            map_location=T.current_device()
        ))
    return vae
