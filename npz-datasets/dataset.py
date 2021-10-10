from deeprobust.graph.data import Dataset
import os.path as osp
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, root, name, setting='gcn', seed=None, require_mask=False):
        '''
        Adopted from https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/data/dataset.py
        '''
        self.name = name.lower()
        self.setting = setting.lower()

        self.seed = seed
        self.url = None
        self.root = osp.expanduser(osp.normpath(root))
        self.data_folder = osp.join(root, self.name)
        self.data_filename = self.data_folder + '.npz'
        # Make sure dataset file exists
        assert osp.exists(self.data_filename), f"{self.data_filename} does not exist!" 
        self.require_mask = require_mask

        self.require_lcc = True if setting == 'nettack' else False
        self.adj, self.features, self.labels = self.load_data()
        self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test()
        if self.require_mask:
            self.get_mask()
    
    def get_adj(self):
        adj, features, labels = self.load_npz(self.data_filename)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1

        if self.require_lcc:
            lcc = self.largest_connected_components(adj)

            # adj = adj[lcc][:, lcc]
            adj_row = adj[lcc]
            adj_csc = adj_row.tocsc()
            adj_col = adj_csc[:, lcc]
            adj = adj_col.tolil()

            features = features[lcc]
            labels = labels[lcc]
            assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

        return adj, features, labels
    
    def get_train_val_test(self):
        if self.setting == "exist":
            with np.load(self.data_filename) as loader:
                idx_train = loader["idx_train"]
                idx_val = loader["idx_val"]
                idx_test = loader["idx_test"]
            return idx_train, idx_val, idx_test
        else:
            return super().get_train_val_test()