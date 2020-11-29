import numpy as np
import argparse, pkgutil, importlib, contextlib, os
from ._dataset import TransformSPAdj
random_state = np.random.RandomState() #pylint: disable=no-member

def set_random_seed(seed):
    global random_state
    random_state = np.random.RandomState(seed) #pylint: disable=no-member

def add_subparsers(parser:argparse.ArgumentParser):
    dataset_list = [modname for importer, modname, ispkg in pkgutil.iter_modules(path=__path__)
        if not modname.startswith("_")]
    parser.add_argument("datafmt", choices=dataset_list, help="Dataset selected for experiment")
    try:
        with contextlib.redirect_stderr(os.devnull):
            known_args, _ = parser.parse_known_args()
        dataset_name = known_args.datafmt
        dataset = importlib.import_module("."+dataset_name, package=__name__)
        if hasattr(dataset, "add_subparser_args"):
            dataset.add_subparser_args(parser)
    except:
        pass
