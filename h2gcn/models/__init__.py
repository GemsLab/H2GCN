import tensorflow as tf
import numpy as np
from modules import logger
import argparse
import pkgutil
import importlib
import contextlib
import os
import json
import re
from pathlib import Path
from enum import Enum, auto
from . import _layers as layers


def add_subparsers(parser: argparse.ArgumentParser):
    model_list = [modname for importer, modname, ispkg in pkgutil.iter_modules(path=__path__)
                  if not modname.startswith("_")]
    parser.add_argument("model", choices=model_list,
                        help="Network model selected for experiment")
    try:
        with contextlib.redirect_stderr(os.devnull):
            known_args, _ = parser.parse_known_args()
        model_name = known_args.model
    except:
        pass
    else:
        model = importlib.import_module("."+model_name, package=__name__)
        if hasattr(model, "add_subparser_args"):
            model.add_subparser_args(parser)
            print(f"Using model: {model}")


class Layer:
    DENSE = "F"
    DROPOUT = "D"
    GCN = "G"
    RELU = "R"
    CONCAT = "C"
    VECTORIZE = "V"
    IDENTITY = "I"
    SLICE = "S"
    EXPERIMENTAL = "X"
    LAMBDA = "lambda"


def parse_network_setup(network_setup_str, output_dim,
                        _dense_units=None, _dropout_rate=None, 
                        parse_preprocessing=False):
    network_setup = re.split(r"-(?![^[]*\])", network_setup_str)
    networkConf = []
    embeddingDefined = False
    for layer_str in network_setup:
        if layer_str[0] == "[" and layer_str[-1] == "]":
            layer_str = layer_str[1:-1].strip()
        
        if layer_str.startswith("lambda"):
            networkConf.append(
                (Layer.LAMBDA, {"lambda": layer_str})
            )
        elif layer_str[0] in ["F", "M"]:
            kwargs = dict()
            if len(layer_str[1:]) > 0:
                if layer_str[1:] == "O":
                    num_hidden_units = output_dim
                    kwargs["beginOutput"] = True
                else:
                    num_hidden_units = int(layer_str[1:])
            else:
                assert _dense_units is not None
                num_hidden_units = _dense_units
            if layer_str[0] == "F":
                networkConf.append(
                    (Layer.DENSE, dict(units=num_hidden_units, use_bias=True, **kwargs)))
            else:
                networkConf.append(
                    (Layer.DENSE, dict(units=num_hidden_units, use_bias=False, **kwargs)))
        elif layer_str[0] == "D":
            if len(layer_str[1:]) > 0:
                dropout_rate = float(layer_str[1:])
            else:
                assert _dropout_rate is not None
                dropout_rate = _dropout_rate
            networkConf.append(
                (Layer.DROPOUT, dict(dropout_rate=dropout_rate)))
        elif layer_str[0] == "G":
            if len(layer_str) > 1:
                hopInds = set([int(i) for i in layer_str[1:].split("_")])
            else:
                hopInds = None
            networkConf.append(
                (Layer.GCN, dict(hops = hopInds))
            )
        elif layer_str[0] == "C":
            tags = [i for i in layer_str[1:].split("_")]
            networkConf.append(
                (Layer.CONCAT, dict(tags=tags, addInputs=True))
            )
        elif layer_str[0] == "R":
            networkConf.append(
                (Layer.RELU, dict())
            )
        elif layer_str[0] == "V":
            networkConf.append(
                (Layer.VECTORIZE, dict())
            )
        elif layer_str[0] == "I":
            networkConf.append(
                (Layer.IDENTITY, dict())
            )
        elif layer_str[0] == "S":
            sliceConf = re.search(r"^S([^_]*)(?:_|$)((?:[^_]*(?:_|$))*)", layer_str)
            sliceTag = sliceConf.group(1)
            if not sliceTag:
                sliceTag = None
            if sliceConf.group(2):
                sliceRange = [(int(x) if x else None) for x in sliceConf.group(2).split("_")]
                sliceObj = slice(*sliceRange)
            else:
                sliceObj = slice(None)
            networkConf.append(
                (Layer.SLICE, dict(
                    loadTag=sliceTag,
                    sliceObj=sliceObj
                )))
        elif layer_str[0] == "X":
            xNameConfRe = re.search(r"X([^_]*)(?:_|$)(.*)", layer_str)
            xName = xNameConfRe.group(1)
            xConf = xNameConfRe.group(2)
            networkConf.append(
                (Layer.EXPERIMENTAL, dict(
                    name=xName,
                    conf=xConf,
                    output_dim=output_dim
                ))
            )
        # Modifiers
        elif layer_str[0] == "E":
            assert not embeddingDefined
            networkConf[-1][-1]["isEmbedding"] = True
            embeddingDefined = True
        elif layer_str[0] == "L":
            networkConf[-1][-1]["supervised"] = True
        elif layer_str[0] == "T":
            networkConf[-1][-1]["tag"] = layer_str[1:]
        
        else:
            raise ValueError(
                f"Unknown layer config {layer_str} in network config {network_setup}")
    return networkConf


def toNumpy(x):
    if type(x) is tf.SparseTensor:
        return {
            "indices": x.indices.numpy(),
            "values": x.values.numpy(),
            "dense_shape": x.dense_shape.numpy()
        }
    else:
        return x.numpy()
