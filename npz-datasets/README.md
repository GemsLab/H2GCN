# Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs

## Additional README for Synthetic Datasets in `npz` Format

This README provides information in addition to [the main README](/..) for usages of the two synthetic datasets `syn-cora` and `syn-products` in `npz` format. The data format is extended from the data format of the [`DeepRobust`](https://github.com/DSE-MSU/DeepRobust) library by Li et al.

Note that the new `npz` format does NOT keep the same training, validation and test splits **and ratios** used in our experiments; for replicating our experiments, please follow the approach in [the main README](/..) to download the datasets in the old format.

### Download Datasets

The datasets can be downloaded using the bash scripts provided in the `scripts` folder under this folder.

### Example Usage

The easiset way to make use of these datasets is to first install [`DeepRobust`](https://github.com/DSE-MSU/DeepRobust) (tested on version 0.2.1) and then utilize the `CustomDataset` class in `dataset.py` in this folder.

```python
#!/usr/bin/env python3
from dataset import CustomDataset

# Load the dataset in file `syn-cora/h0.00-r1.npz`
# `seed` controls the generation of training, validation and test splits
dataset = CustomDataset(root="syn-cora", name="h0.00-r1", setting="gcn", seed=15)

adj = dataset.adj # Access adjacency matrix
features = dataset.features # Access node features
```

Refer to [DeepRobust docs](https://deeprobust.readthedocs.io/en/latest/source/deeprobust.graph.data.html#deeprobust.graph.data.Dataset) for more information on the usage of the base class `Dataset` in `dataset.py`. You may also load the data directly from the `npz` files using `numpy`.

### Citation

Please cite our work if you make use of these datasets in your own work:

```bibtex
@article{zhu2020beyond,
  title={Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs},
  author={Zhu, Jiong and Yan, Yujun and Zhao, Lingxiao and Heimann, Mark and Akoglu, Leman and Koutra, Danai},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
