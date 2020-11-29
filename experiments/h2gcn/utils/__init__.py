from .dataset import PlanetoidData, get_cora, TransformAdj, GeomGCNData
from . import signac_tools
from . import dataset
from . import sparsegraph
import os
from contextlib import contextmanager

@contextmanager
def chdir(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)