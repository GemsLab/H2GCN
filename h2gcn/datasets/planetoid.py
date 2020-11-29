from . import *
import tensorflow as tf
from tensorflow import keras
from ._dataset import PlanetoidData, TransformAdj, TransformSPAdj, sp

def add_subparser_args(parser):
    subparser = parser.add_argument_group("Planetoid Format Data Arguments (datasets/planetoid.py)")
    subparser.add_argument("--dataset", type=str, required=True)
    subparser.add_argument("--dataset_path", type=str, dest="_dataset_path", required=True)
    subparser.add_argument("--val_size", type=int, default=500)
    subparser.add_argument("--feature_configs", choices=["no_test", "identity", "labels"], nargs="*", default=[])
    parser.function_hooks["argparse"].appendleft(argparse_callback)

def argparse_callback(args):
    if args.val_size < 0:
        args.val_size = None
    dataset = PlanetoidData(args.dataset, args._dataset_path, val_size=args.val_size)
    for config in args.feature_configs:
        if config == "no_test":
            lil_features = dataset.features.tolil()
            lil_features[dataset.test_mask, :] = 0
            dataset.features = lil_features.tocsr()
        elif config == "identity":
            dataset.set_identity_features()
        elif config == "labels":
            dataset.set_label_one_hot_features()
    args.objects["dataset"] = dataset
    print(f"===> Dataset loaded: {args.dataset}")

