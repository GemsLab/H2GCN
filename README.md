# Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs

Jiong Zhu, Yujun Yan, Lingxiao Zhao, Mark Heimann, Leman Akoglu, and Danai Koutra. 2020. *Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs*. Advances in Neural Information Processing Systems 33 (2020).

[[Paper]](https://arxiv.org/abs/2006.11468)
[[Poster]](https://www.jiongzhu.net/assets/files/F20-Jiong-H2GCN-NeurIPS-Poster.pdf)
[[Slides]](https://www.jiongzhu.net/assets/files/F20-Jiong-H2GCN-NeurIPS-Talk.pdf)

## Updates

- Oct. 2021: We additionally provide our synthetic datasets `syn-cora` and `syn-products` in a more straight-forward `npz` format; see README in folder `/npz-datasets` for more details.

- Aug. 2021: In a [blog post](https://www.jiongzhu.net/revisiting-heterophily-GNNs/), we revisit the problem of heterophily for GNNs and discuss the reasons behind seemly different takeaways in light of recent works in this area.

## Requirements

### Basic Requirements

- **Python** >= 3.7 (tested on 3.8)
- **signac**: this package utilizes [signac](https://signac.io) to manage experiment data and jobs. signac can be installed with the following command:

  ```bash
  pip install signac==1.1 signac-flow==0.7.1 signac-dashboard
  ```

  Note that the latest version of signac may cause incompatibility.
- **numpy** (tested on 1.18.5)
- **scipy** (tested on 1.5.0)
- **networkx** >= 2.4 (tested on 2.4)
- **scikit-learn** (tested on 0.23.2)

### For `H2GCN`

- **TensorFlow** >= 2.0 (tested on 2.2)

Note that it is possible to use `H2GCN` without `signac` and `scikit-learn` on your own data and experimental framework.

### For baselines

We also include the code for the baseline methods in the repository. These code are mostly the same as the reference implementations provided by the authors, *with our modifications* to add JK-connections, interoperability with our experimental pipeline, etc. For the requirements to run these baselines, please refer to the instructions provided by the original authors of the corresponding code, which could be found in each folder under `/baselines`.

As a general note, TensorFlow 1.15 can be used for all code requiring TensorFlow 1.x; for PyTorch, it is usually fine to use PyTorch 1.6; all code should be able to run under Python >= 3.7. In addition, the [basic requirements](#basic-requirements) must also be met.

## Usage

### Download Datasets

The datasets can be downloaded using the bash scripts provided in `/experiments/h2gcn/scripts`, which also prepare the datasets for use in our experimental framework based on `signac`.

We make use of `signac` to index and manage the datasets: the datasets and experiments are stored in hierarchically organized signac jobs, with the **1st level** storing different graphs, **2nd level** storing different sets of features, and **3rd level** storing different training-validation-test splits. Each level contains its own state points and job documents to differentiate with other jobs.

Use `signac schema` to list all available properties in graph state points; use `signac find` to filter graphs using properties in the state points:

```bash
cd experiments/h2gcn/

# List available properties in graph state points
signac schema

# Find graphs in syn-products with homophily level h=0.1
signac find numNode 10000 h 0.1

# Find real benchmark "Cora"
signac find benchmark true datasetName\.\$regex "cora"
```

`/experiments/h2gcn/utils/signac_tools.py` provides helpful functions to iterate through the data space in Python; more usages of signac can be found in these [documents](https://docs.signac.io/en/latest/).

#### Alternative: Download `syn-cora` and `syn-products` in `npz` Format

If you are interested in using the two synthetic datasets `syn-cora` and `syn-products` in your own research, we additionally provide them in a more straight-forward `npz` format; See README in folder `/npz-datasets` for more details. Note that the new `npz` format does NOT keep the same training, validation and test splits **and ratios** used in our experiments; for [replicating our experiments](#replicate-experiments-with-signac), please follow the above approach to download the datasets in the old format.

### Replicate Experiments with `signac`

- To replicate our experiments of each model on specific datasets, use Python scripts in `/experiments/h2gcn`, and the corresponding JSON config files in `/experiments/h2gcn/configs`. For example, to run `H2GCN` on our synthetic benchmarks `syn-cora`:

  ```bash
  cd experiments/h2gcn/
  python run_hgcn_experiments.py -c configs/syn-cora/h2gcn.json [-i] run [-p PARALLEL_NUM]
  ```

  - Files and results generated in experiments are also stored with signac on top of the hierarchical order introduced above: the **4th level** separates different models, and the **5th level** stores files and results generated in different runs with different parameters of the same model.
  - By default, `stdout` and `stderr` of each model are stored in `terminal_output.log` in the 4th level; use `-i` if you want to see them through your terminal.
  - Use `-p` if you want to run experiments in parallel on multiple graphs (1st level).
  - Baseline models can be run through the following scripts:

    - **GCN, GCN-Cheby, GCN+JK and GCN-Cheby+JK**: `run_gcn_experiments.py`
    - **GraphSAGE, GraphSAGE+JK**: `run_graphsage_experiments.py`
    - **MixHop**: `run_mixhop_experiments.py`
    - **GAT**: `run_gat_experiments.py`
    - **MLP**: `run_hgcn_experiments.py`
  
- To summarize experiment results of each model on specific datasets to a CSV file, use Python script `/experiments/h2gcn/run_experiments_summarization.py` with the corresponding model name and config file. For example, to summarize `H2GCN` results on our synthetic benchmark `syn-cora`:

  ```bash
  cd experiments/h2gcn/
  python run_experiments_summarization.py h2gcn -f configs/syn-cora/h2gcn.json
  ```

- To list all paths of the 3rd level datasets splits used in a experiment (in planetoid format) without running experiments, use the following command:

  ```bash
  cd experiments/h2gcn/
  python run_hgcn_experiments.py -c configs/syn-cora/h2gcn.json --check_paths run
  ```

### Standalone H2GCN Package

Our implementation of H2GCN is stored in the `h2gcn` folder, which can be used as a standalone package on your own data and experimental framework.

Example usages:

- H2GCN-2

  ```bash
  cd h2gcn
  python run_experiments.py H2GCN planetoid \
    --dataset ind.citeseer \
    --dataset_path ../baselines/gcn/gcn/data/
  ```

- H2GCN-1

  ```bash
  cd h2gcn
  python run_experiments.py H2GCN planetoid \
    --network_setup M64-R-T1-G-V-C1-D0.5-MO \
    --dataset ind.citeseer \
    --dataset_path ../baselines/gcn/gcn/data/
  ```

- Use `--help` for more advanced usages:

  ```bash
  python run_experiments.py H2GCN planetoid --help
  ```

We only support datasets stored in [`planetoid` format](https://github.com/kimiyoung/planetoid). You could also add support to different data formats and models beyond H2GCN by adding your own modules to `/h2gcn/datasets` and `/h2gcn/models`, respectively; check out ou code for more details.

## Contact

Please contact Jiong Zhu (jiongzhu@umich.edu) in case you have any questions.

## Citation

Please cite our paper if you make use of this code in your own work:

```bibtex
@article{zhu2020beyond,
  title={Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs},
  author={Zhu, Jiong and Yan, Yujun and Zhao, Lingxiao and Heimann, Mark and Akoglu, Leman and Koutra, Danai},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
