import signac
import hashlib
import json
import shutil
import pickle
import os
from pathlib import Path
import numpy as np
import scipy.sparse
from . import graphgen
from collections import OrderedDict
import networkx as nx
import utils
import threading

random_state = np.random.RandomState()  # pylint: disable=no-member


def get_class_indices(ally, classID):
    return np.nonzero(ally[:, classID] == 1)[0]


def row_sample(ally, source_graph):
    classSize = np.sum(ally, axis=0)
    allx = np.zeros((len(ally), source_graph.feature_dim))
    syn_cls_list = np.argsort(classSize)[::-1]
    for source_cls, syn_cls in zip(source_graph.sort_label_by_size(), syn_cls_list):
        feature_iter = source_graph.feature_iter(source_cls)
        syn_node_list = get_class_indices(ally, syn_cls)
        random_state.shuffle(syn_node_list)
        for feature, syn_node in zip(feature_iter, syn_node_list):
            allx[syn_node, :] = feature.toarray()
    return allx

# # Test code for row_sample
# if __name__ == "__main__":
#     source_graph = utils.get_cora()
#     ally = np.array([
#         [1, 0, 0],
#         [0, 1, 0],
#         [1, 0, 0],
#         [0, 0, 1],
#         [0, 0, 1],
#         [1, 0, 0]
#     ])
#     allx = row_sample(ally, source_graph)
#     assert allx.shape == (6, 1433)
#     dot_result = np.zeros((ally.shape[0], ally.shape[0]))
#     for ix, iy in np.ndindex(dot_result.shape):
#         dot_result[ix, iy] = np.dot(allx[ix, :], allx[iy, :])
#     print(dot_result)

ogbnLockDict = dict()
def ogbn_generate_split(job: signac.Project.Job, splitJob: signac.Project.Job,
                        feature_graph_name, feature_graph_files):
    import constraint
    with utils.chdir(splitJob.sp.ogbn_path):
        from ogb.nodeproppred import NodePropPredDataset
        d_name = splitJob.sp.ogbn_name
        
        lock = ogbnLockDict.setdefault(splitJob.sp.ogbn_path, threading.Lock())
        if not os.path.exists("dataset"): # In case dataset is not downloaded
            lock.acquire()
            ogbnDataset = NodePropPredDataset(name = d_name)
            lock.release()
        else:
            ogbnDataset = NodePropPredDataset(name = d_name)
        
        split_idx = ogbnDataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = ogbnDataset[0]
    
    with job:
        splitJobSrc = utils.signac_tools.access_proj_job(
            job, splitJob.sp.feature_source, splitJob.sp.split_source
        )
        splitSrcName = splitJobSrc.doc["split_name"]
        # Copy not changing files
        for source_file, dest_file in [
            (splitJobSrc.fn(f"{splitSrcName}.{ext}"),
             splitJob.fn(f"{feature_graph_name}.{ext}"))
            for ext in ('y', 'ty', 'ally', 'graph', 'test.index')
        ]:
            shutil.copy2(source_file, dest_file)
        
        
        with splitJobSrc:
            datasetSrc = utils.PlanetoidData(splitJobSrc.doc.split_name, ".", 
                val_size=None)
        
        ogbnLabelCount = np.zeros((3, ogbnDataset.num_classes))
        ogbnLabelCount[0, :] = (label[train_idx] == np.arange(ogbnDataset.num_classes)).sum(0)
        ogbnLabelCount[1, :] = (label[valid_idx] == np.arange(ogbnDataset.num_classes)).sum(0)
        ogbnLabelCount[2, :] = (label[test_idx] == np.arange(ogbnDataset.num_classes)).sum(0)
        
        srcLabelCount = np.zeros((3, job.sp.numClass))
        srcLabelCount[0, :] = datasetSrc.y_all[datasetSrc.train_mask, :].sum(0)
        srcLabelCount[1, :] = datasetSrc.y_all[datasetSrc.val_mask, :].sum(0)
        srcLabelCount[2, :] = datasetSrc.y_all[datasetSrc.test_mask, :].sum(0)
        
        problem = constraint.Problem()
        problem.addVariables(range(job.sp.numClass), range(ogbnDataset.num_classes))
        problem.addConstraint(constraint.AllDifferentConstraint())
        for i in range(job.sp.numClass):
            problem.addConstraint(lambda x: np.all(ogbnLabelCount[:, x] >= srcLabelCount[:, i]), (i,))
        solution = problem.getSolution()

        for srcClass, dstClass in solution.items():
            assert np.all(ogbnLabelCount[:, dstClass] >= srcLabelCount[:, srcClass])

        newFeatures = np.zeros((datasetSrc.num_samples, graph["node_feat"].shape[1]))
        for scope, idx in (("train", train_idx), ("val", valid_idx), ("test", test_idx)):
            scope_mask = getattr(datasetSrc, f"{scope}_mask")
            for srcClass, dstClass in solution.items():
                srcOpMask = np.logical_and(scope_mask, datasetSrc.labels == srcClass)
                dstSampleSet = list(set(idx).intersection(np.where(label == dstClass)[0]))
                sampleInds = random_state.choice(dstSampleSet, srcOpMask.sum(), replace=False)
                newFeatures[srcOpMask, :] = graph["node_feat"][sampleInds, :]

        x_mask = datasetSrc.train_mask
        allx_mask = (datasetSrc.train_mask + datasetSrc.val_mask)
        test_mask = datasetSrc.test_mask

        x = newFeatures[x_mask]
        allx = newFeatures[allx_mask]
        tx = newFeatures[test_mask]

        # .x; .tx; .allx
        pickle.dump(scipy.sparse.csr_matrix(x), open(
            splitJob.fn(f"{feature_graph_name}.x"), "wb"))
        pickle.dump(scipy.sparse.csr_matrix(allx), open(
            splitJob.fn(f"{feature_graph_name}.allx"), "wb"))
        pickle.dump(scipy.sparse.csr_matrix(tx), open(
            splitJob.fn(f"{feature_graph_name}.tx"), "wb"))

        assert all(map(splitJob.isfile, feature_graph_files))
        splitJob.doc["succeeded"] = True
        splitJob.doc["split_name"] = feature_graph_name
        splitJob.doc.val_size = splitJobSrc.doc.val_size


def load_feature_file(feature_file):
    if feature_file.endswith(".npz"):
        allx = scipy.sparse.load_npz(feature_file).todense()
    else:
        allx = np.load(feature_file)
    return allx


def select_indices(mode, sampled_ind, graph, ally, num_classes):
    if mode.endswith("c"):  # Sample per class
        train_size = int(mode[:-1])
        if len(graph) < train_size * num_classes:
            return None
        train_indices = np.zeros(train_size * num_classes, dtype=np.int64) - 1
        for cls_i in range(num_classes):
            example_indices = np.nonzero(
                np.logical_and(ally[:, cls_i] == 1,
                               np.logical_not(sampled_ind))
            )[0]
            if len(example_indices) < train_size:
                return None
            train_indices_cls = random_state.choice(
                example_indices, train_size, replace=False)
            train_indices[train_size * cls_i: train_size *
                          (cls_i + 1)] = train_indices_cls
            sampled_ind[train_indices_cls] = True  # Mark already sampled items
    elif mode.endswith("p"):  # Sample per class ratio
        train_ratio = float(mode[:-1])
        train_indices = []
        for cls_i in range(num_classes):
            example_indices = np.nonzero(
                np.logical_and(ally[:, cls_i] == 1,
                               np.logical_not(sampled_ind))
            )[0]
            train_indices_cls = random_state.choice(
                example_indices, np.math.floor(
                    train_ratio * (ally[:, cls_i] == 1).sum()),
                replace=False)
            sampled_ind[train_indices_cls] = True  # Mark already sampled items
            train_indices += list(train_indices_cls)
        train_indices = np.array(train_indices)
    elif mode == "":
        train_indices = np.nonzero(np.logical_and(
            np.logical_not(sampled_ind), ally.sum(1) > 0))[0]
        sampled_ind[train_indices] = True
    else:  # Sampling regardless of class
        train_size = int(mode)
        assert len(graph) >= train_size
        example_indices = np.nonzero(np.logical_and(
            np.logical_not(sampled_ind), ally.sum(1) > 0))[0]
        train_indices = random_state.choice(
            example_indices, train_size, replace=False)
        sampled_ind[train_indices] = True
    return train_indices


def generate_split(job: signac.Project.Job, graph, ally, G, feature_file,
                   splitJob, feature_graph_name, feature_graph_files,
                   train_indices=None, test_indices=None, validation_indices=None):
    allx = load_feature_file(feature_file)
    num_classes = ally.shape[1]
    node_mapping = dict()
    sampled_ind = np.zeros(ally.shape[0]).astype(bool)
    words = splitJob.sp.split_config.split("_")
    # Select training instances
    if train_indices is None:
        train_indices = select_indices(
            words[0], sampled_ind, graph, ally, num_classes)
    else:
        assert not np.any(sampled_ind[train_indices])
        sampled_ind[train_indices] = True
    
    if train_indices is None:
        print(
            f"[generate_split@{job.get_id()}] Insufficient samples for split {splitJob.sp.split_config}")
        splitJob.doc.disabled = True
        return

    assert np.all(train_indices >= 0)
    random_state.shuffle(train_indices)
    train_indices = train_indices.astype(int)
    for i, node in enumerate(train_indices):
        node_mapping[node] = i
    x = allx[train_indices, :]
    y = ally[train_indices, :]

    if words[1] != "" and words[2] == "":
        order = ["validation", "test"]
    elif words[1] == "" and words[2] == "":
        raise ValueError(f"Unsupported split config {splitJob.sp.split_config}")
    else:
        order = ["test", "validation"]
    
    for scope in order:
        if scope == "test":
            word = words[2]
            indices = test_indices
        elif scope == "validation":
            word = words[1]
            indices = validation_indices
        else:
            raise ValueError()

        if indices is None:
            indices = select_indices(
                word, sampled_ind, graph, ally, num_classes)
        else:
            assert not np.any(sampled_ind[indices])
            sampled_ind[indices] = True
        
        if scope == "test":
            test_indices = indices
        elif scope == "validation":
            validation_indices = indices
        else:
            raise ValueError()

    if test_indices is None:
        print(
            f"[generate_split@{job.get_id()}] Insufficient samples for split {splitJob.sp.split_config}")
        splitJob.doc.disabled = True
        return
    tx = allx[test_indices, :]
    ty = ally[test_indices, :]

    new_allx = np.vstack((x, allx[validation_indices, :]))
    new_ally = np.vstack((y, ally[validation_indices, :]))
    splitJob.doc.val_size = len(validation_indices)

    for node in validation_indices:
        node_mapping[node] = len(node_mapping)

    # Map remaining instances
    if not np.all(sampled_ind):
        wild_indices = np.nonzero(np.logical_not(sampled_ind))[0]
        for node in wild_indices:
            node_mapping[node] = len(node_mapping)
        new_allx = np.vstack((new_allx, allx[wild_indices, :]))
        new_ally = np.vstack((new_ally, ally[wild_indices, :]))

    word_ptr = 3
    
    # Write files
    with job:
        # .test.index
        with open(splitJob.fn(feature_graph_name + ".test.index"), "w") as test_index_f:
            for node in test_indices:
                test_index_f.write("{}\n".format(len(node_mapping)))
                node_mapping[node] = len(node_mapping)

        # .graph, .gpickle.gz
        G_new = nx.relabel_nodes(G, node_mapping, copy=True)
        generator = graphgen.GraphGenerator(job.sp.numClass)
        generator.save_graph(G_new, splitJob.workspace(), feature_graph_name)
        generator.save_nx_graph(
            G_new, splitJob.workspace(), feature_graph_name)

        # .y; .ty; .ally
        pickle.dump(y, open(splitJob.fn(feature_graph_name + ".y"), "wb"))
        pickle.dump(ty, open(splitJob.fn(feature_graph_name + ".ty"), "wb"))
        pickle.dump(new_ally, open(splitJob.fn(
            feature_graph_name + ".ally"), "wb"))

        # .x; .tx; .allx
        pickle.dump(scipy.sparse.csr_matrix(x), open(
            splitJob.fn(feature_graph_name + ".x"), "wb"))
        pickle.dump(scipy.sparse.csr_matrix(tx), open(
            splitJob.fn(feature_graph_name + ".tx"), "wb"))
        pickle.dump(scipy.sparse.csr_matrix(new_allx), open(
            splitJob.fn(feature_graph_name + ".allx"), "wb"))

    assert all(map(splitJob.isfile, feature_graph_files))
    splitJob.data["node_mapping"] = json.dumps({int(x): y for x, y in node_mapping.items()})
    splitJob.doc["succeeded"] = True
    splitJob.doc["split_name"] = feature_graph_name
