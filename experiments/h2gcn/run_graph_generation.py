import signac
from flow import FlowProject
import modules.graphgen as graphgen
import numpy as np
import networkx as nx
import pickle
import itertools
import os
import shutil
import random
import hashlib
import json
import sys
import scipy.sparse
import modules.feature_generation as feature_generation
import modules.graph_stats as graph_stats
import utils
from utils import sparsegraph
from pathlib import Path

random_state = np.random.RandomState()  # type: np.random.RandomState


def reset_random_state(job: signac.Project.Job, seed=None):
    global random_state
    if seed is None:
        seed = job.get_id()
    np_seed = random.Random(seed).randrange(0, 2**32)
    random_state = np.random.RandomState(np_seed)
    print(f"[{job.get_id()}] Random seed is set to {np_seed}")
    return random_state


@FlowProject.label
def graph_generated(job):
    return job.isfile(job.sp.graphName + ".graph") \
        and job.isfile(job.sp.graphName + ".ally") \
        and job.isfile(job.sp.graphName + ".gpickle.gz")


@FlowProject.operation
@FlowProject.post(graph_generated)
def generate_graph(job: signac.Project.Job):
    print("Generating graph for job {}".format(job.get_id()))
    graphgen.random_state = reset_random_state(job)
    if job.sp.method == "mixhop":
        generator = graphgen.MixhopGraphGenerator(
            job.sp.classRatio, job.sp.heteroClsWeight, heteroWeightsExponent=job.sp.heteroWeightsExponent)
        G = generator(job.sp.numNode, job.sp.m, job.sp.m0, job.sp.h)
        generator.save_graph(G, job.workspace(), job.sp.graphName)
        generator.save_y(G, job.workspace(), job.sp.graphName)
        generator.save_nx_graph(G, job.workspace(), job.sp.graphName)

    elif job.sp.method == "planetoid":
        with job:
            dataset = utils.PlanetoidData(job.sp.datasetName, "data_source")
        G = dataset.getNXGraph()
        generator = graphgen.GraphGenerator(job.sp.numClass)
        generator.save_graph(G, job.workspace(), job.sp.graphName)
        generator.save_y(G, job.workspace(), job.sp.graphName)
        generator.save_nx_graph(G, job.workspace(), job.sp.graphName)

        featureProject = utils.signac_tools.getFeatureProject(job)
        featureJob = featureProject.open_job({
            "feature_type": "unmodified"
        }).init()

        splitProject = utils.signac_tools.getSplitProject(featureJob)
        trainSetSize = dataset.y_all[dataset.train_mask].sum(0)
        if len(np.unique(trainSetSize)) == 1:
            trainSetSize = "{}c".format(int(trainSetSize[0]))
        else:
            trainSetSize = int(dataset.train_mask.sum())
        splitJob = splitProject.open_job({
            "split_config": "{}__{}".format(trainSetSize, int(dataset.test_mask.sum()))
        }).init()
    elif job.sp.method == "GeomGCN":
        with job:
            dataset = utils.GeomGCNData(job.sp.datasetName, "data_source")
        G = dataset.getNXGraph()
        generator = graphgen.GraphGenerator(job.sp.numClass)
        generator.save_graph(G, job.workspace(), job.sp.graphName)
        generator.save_y(G, job.workspace(), job.sp.graphName)
        generator.save_nx_graph(G, job.workspace(), job.sp.graphName)

        featureProject = utils.signac_tools.getFeatureProject(job)
        featureJob = featureProject.open_job({
            "feature_type": "unmodified"
        }).init()

        output_name = f"{job.sp.graphName}-unmodified.allx.npz"
        allx = dataset.features
        allx = scipy.sparse.csr_matrix(allx)
        scipy.sparse.save_npz(featureJob.fn(output_name), allx)

        featureJob.doc["feature_file"] = output_name
        featureJob.doc["feature_name"] = f"{job.sp.datasetName}-unmodified"
        featureJob.doc["succeeded"] = True
    elif job.sp.method == "SparseGraph":
        with job:
            spgraph = sparsegraph.io.load_dataset(
                str(Path("data_source")/job.sp.datasetName))

        for command in job.sp.get("preprocess", []):
            exec(command)

        G = spgraph.getNXGraph()
        generator = graphgen.GraphGenerator(job.sp.numClass)
        generator.save_graph(G, job.workspace(), job.sp.graphName)
        generator.save_y(G, job.workspace(), job.sp.graphName)
        generator.save_nx_graph(G, job.workspace(), job.sp.graphName)

        featureProject = utils.signac_tools.getFeatureProject(job)
        featureJob = featureProject.open_job({
            "feature_type": "unmodified"
        }).init()

        if spgraph.attr_matrix is not None:
            # Generate features
            output_name = f"{job.sp.graphName}-unmodified.allx.npz"
            allx = spgraph.attr_matrix
            allx = scipy.sparse.csr_matrix(allx)
            scipy.sparse.save_npz(featureJob.fn(output_name), allx)

            featureJob.doc["feature_file"] = output_name
            featureJob.doc["feature_name"] = f"{job.sp.datasetName}-unmodified"
            featureJob.doc["succeeded"] = True

    elif job.sp.method == "copy":
        graph_path, ally_path, ty_path, test_index_path = map(
            lambda x: job.fn("source_graph/{}{}".format(job.sp.source_name, x)), (".graph", ".ally", ".ty", ".test.index"))
        graph = pickle.load(open(graph_path, "rb"))
        G = nx.from_dict_of_lists(graph)
        ally = np.load(ally_path, allow_pickle=True)
        ty = np.load(ty_path, allow_pickle=True)

        attrs = dict()
        for i in range(ally.shape[0]):
            color = np.nonzero(ally[i, :])[0] + 1
            assert len(color) == 1, print(i, color)
            color = color[0]
            attrs[i] = {"color": color}

        for i, line in enumerate(open(test_index_path, "r")):
            node_id = int(line.strip())
            color = np.nonzero(ty[i, :])[0] + 1
            assert len(color) == 1, print(i, color)
            color = color[0]
            attrs[node_id] = {"color": color}

        assert i == ty.shape[0] - 1
        assert len(attrs) == len(G.node)
        nx.set_node_attributes(G, attrs)

        generator = graphgen.GraphGenerator(job.sp.numClass)
        generator.save_graph(G, job.workspace(), job.sp.graphName)
        generator.save_y(G, job.workspace(), job.sp.graphName)
        generator.save_nx_graph(G, job.workspace(), job.sp.graphName)
    else:
        raise ValueError("Unknown generation method {}".format(job.sp.method))


@FlowProject.label
def statistics_calculated(job: signac.Project.Job):
    check_doc = all(
        ((stat_key in job.doc) and (graph_stats.stats_dict[stat_key][1] <= 1))
        for stat_key in graph_stats.stats_dict if graph_stats.stats_dict[stat_key][1]
    )

    check_data = all(
        ((stat_key in job.data) and (graph_stats.stats_dict[stat_key][2] <= 1))
        for stat_key in graph_stats.stats_dict if graph_stats.stats_dict[stat_key][2]
    )
    return (check_data and check_doc)


@FlowProject.operation
@FlowProject.pre.after(generate_graph)
@FlowProject.post(statistics_calculated)
def calculate_statistics(job: signac.Project.Job):
    G = nx.read_gpickle(
        job.fn(job.sp.graphName + ".gpickle.gz"))  # type: nx.Graph
    ally = pickle.load(
        open(job.fn(job.sp.graphName + ".ally"), "rb"), encoding="bytes")
    for stat_key, (stat_func, addToDoc, addToData) in graph_stats.stats_dict.items():
        if (addToData and (stat_key not in job.data)) or (addToDoc and (stat_key not in job.doc)) or (addToData > 1) or (addToDoc > 1):
            print(
                f"[calculate_statistics@{job.get_id()}] Calculating {stat_key}...")
            resultDict = stat_func(G=G, ally=ally, job=job)
            assert stat_key in resultDict
            for key, value in resultDict.items():
                addToDocResult = graph_stats.stats_dict[key][1]
                addToDataResult = graph_stats.stats_dict[key][2]
                if addToDocResult:
                    job.doc[key] = value
                if addToDataResult:
                    job.data[key] = value


def feature_file_iter(job: signac.Project.Job):
    for featureJob in utils.signac_tools.feature_iter(job):
        try:
            feature_type = featureJob.sp.feature_type
        except:
            print(job.get_id(), featureJob.get_id())
            raise
        if feature_type == "naive":
            yield featureJob.fn("{}-{}-{}.allx".format(job.sp.graphName, feature_type, featureJob.sp.var_factor))
        elif feature_type == "naive_npz":
            yield featureJob.fn("{}-{}-{}.allx.npz".format(job.sp.graphName, feature_type, featureJob.sp.var_factor))
        elif feature_type == "sample":
            type_str = featureJob.sp.sample_type
            if type_str == "cora_row":
                yield featureJob.fn(f"{job.sp.graphName}-{feature_type}-{type_str}.allx.npz")
            elif type_str in ["ogbn"]:
                pass
            else:
                raise NotImplementedError(
                    f"{job.sp.graphName}-{feature_type}-{type_str}")
        elif feature_type in ["unmodified"]:
            if job.sp.method == "planetoid":
                pass
            elif job.sp.method in ["GeomGCN", "SparseGraph"]:
                yield featureJob.fn(f"{job.sp.graphName}-{feature_type}.allx.npz")
            else:
                raise NotImplementedError(
                    f"{job.sp.graphName}-{job.sp.method}-{feature_type}")
        else:
            raise ValueError("Unknown feature type {}".format(feature_type))


@FlowProject.label
def feature_generated(job: signac.Project.Job):
    result = list(map(os.path.exists, feature_file_iter(job)))
    result_succeeded = list(featureJob.doc.get("succeeded", False)
                            for featureJob in utils.signac_tools.feature_iter(job))
    return len(result_succeeded) > 0 and all(result) and all(result_succeeded)


@FlowProject.operation
@FlowProject.pre(lambda job: len(list(utils.signac_tools.feature_iter(job))) > 0)
@FlowProject.pre.after(generate_graph)
@FlowProject.post(feature_generated)
def generate_feature(job: signac.Project.Job):
    for featureJob in utils.signac_tools.feature_iter(job):
        feature_type = featureJob.sp.feature_type
        if feature_type == "naive":
            type_str = featureJob.sp.var_factor
            output_name = "{}-{}-{}.allx".format(
                job.sp.graphName, feature_type, type_str)
            if featureJob.isfile(output_name):
                print("[generate_feature@{}] {} already exists. Skipping...".format(
                    job.get_id(), output_name))
                continue
            print("[generate_feature@{}] Generating features to {}".format(
                job.get_id(), output_name))
            ally = pickle.load(
                open(job.fn(job.sp.graphName + ".ally"), "rb"), encoding="bytes")
            if type_str == "all":
                allx = ally
            else:
                raise NotImplementedError()
            np.save(open(featureJob.fn(output_name), "wb"), allx)

            featureJob.doc["feature_file"] = output_name
            featureJob.doc["feature_name"] = f"{feature_type}-{type_str}"
            featureJob.doc["succeeded"] = True

        elif feature_type == "naive_npz":
            type_str = featureJob.sp.var_factor
            output_name = "{}-{}-{}.allx.npz".format(
                job.sp.graphName, feature_type, type_str)
            if featureJob.isfile(output_name):
                print("[generate_feature@{}] {} already exists. Skipping...".format(
                    job.get_id(), output_name))
                continue
            print("[generate_feature@{}] Generating features to {}".format(
                job.get_id(), output_name))
            ally = pickle.load(
                open(job.fn(job.sp.graphName + ".ally"), "rb"), encoding="bytes")
            if type_str == "all":
                allx = ally
            elif type_str == "identity":
                allx = np.eye(ally.shape[0])
            else:
                raise NotImplementedError()
            allx = scipy.sparse.csr_matrix(allx)
            scipy.sparse.save_npz(featureJob.fn(output_name), allx)

            featureJob.doc["feature_file"] = output_name
            featureJob.doc["feature_name"] = f"{feature_type}-{type_str}"
            featureJob.doc["succeeded"] = True

        elif feature_type == "sample":
            type_str = featureJob.sp.sample_type
            if type_str == "cora_row":
                output_name = "{}-{}-{}.allx.npz".format(
                    job.sp.graphName, feature_type, type_str)
                if featureJob.isfile(output_name):
                    print("[generate_feature@{}] {} already exists. Skipping...".format(
                        job.get_id(), output_name))
                    continue
                ally = pickle.load(
                    open(job.fn(job.sp.graphName + ".ally"), "rb"), encoding="bytes")
                cora = utils.get_cora()
                classSize = np.sum(ally, axis=0)
                if cora.feature_sample_eligible(classSize):
                    print("[generate_feature@{}] Generating features to {} by row-based cora feature sampling".format(
                        job.get_id(), output_name))
                    feature_generation.random_state = reset_random_state(
                        job, (job.get_id(), output_name))
                    allx = feature_generation.row_sample(ally, cora)
                    allx = scipy.sparse.csr_matrix(allx)
                    scipy.sparse.save_npz(featureJob.fn(output_name), allx)

                    featureJob.doc["feature_file"] = output_name
                    featureJob.doc["succeeded"] = True
                else:
                    featureJob.doc["disabled"] = True
                    featureJob.doc["disable_reason"] = f"[generate_feature@{job.get_id()}] {job.sp.graphName} is ineligible for row-based cora feature sampling"
                    print(featureJob.doc["disable_reason"])
            elif type_str in ["ogbn"]:
                if not featureJob.doc["succeeded"]:
                    raise ValueError(
                        f"[generate_feature@{job.get_id()}] {type_str} feature is not marked as succeeded for job {featureJob.get_id()}")
            else:
                raise NotImplementedError()

        elif feature_type == "unmodified":
            if job.sp.method == "planetoid":

                # This block is incompatibale with what current structure shows.
                output_name = f"{job.sp.datasetName}-{feature_type}.allx.npz"
                with job:
                    dataset = utils.PlanetoidData(
                        job.sp.datasetName, "data_source")
                allx = dataset.features
                allx = scipy.sparse.csr_matrix(allx)
                scipy.sparse.save_npz(featureJob.fn(output_name), allx)

                featureJob.doc["feature_file"] = output_name
                ###

                featureJob.doc["feature_name"] = f"{job.sp.datasetName}-{feature_type}"
                featureJob.doc["succeeded"] = True

            elif job.sp.method == "GeomGCN":
                output_name = f"{job.sp.graphName}-{feature_type}.allx.npz"
                if featureJob.isfile(output_name) and featureJob.doc.get("succeeded", False):
                    print("[generate_feature@{}] {} already exists. Skipping...".format(
                        job.get_id(), output_name))
                    continue
                print("[generate_feature@{}] Write Geom-GCN features to {}".format(
                    job.get_id(), output_name))
                with job:
                    dataset = utils.GeomGCNData(
                        job.sp.datasetName, "data_source")
                allx = dataset.features
                allx = scipy.sparse.csr_matrix(allx)
                scipy.sparse.save_npz(featureJob.fn(output_name), allx)

                featureJob.doc["feature_file"] = output_name
                featureJob.doc["feature_name"] = f"{job.sp.datasetName}-unmodified"
                featureJob.doc["succeeded"] = True
            else:
                raise NotImplementedError(
                    f"{job.sp.graphName}-{job.sp.method}-{feature_type}")
        else:
            raise ValueError("Unknown feature type {}".format(feature_type))


def feature_split_iter(job: signac.Project.Job):
    for featureJob in utils.signac_tools.feature_iter(job):
        feature_file = featureJob.doc.get("feature_file")
        feature_name = featureJob.doc.get("feature_name")
        if feature_file:
            for splitJob in utils.signac_tools.split_iter(featureJob):
                split_config = splitJob.sp.split_config
                ftmp = feature_file.replace(".npz", "")
                feature_graph_name = "{}-{}".format(
                    os.path.splitext(ftmp)[0], split_config)

                splitFormat = splitJob.sp.get("format", "planetoid")
                if splitFormat == "planetoid":
                    feature_graph_files = (f"{feature_graph_name}.{ext}"
                                           for ext in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index'])
                elif splitFormat == "sparsegraph":
                    feature_graph_files = (
                        f"{feature_graph_name}.{ext}" for ext in ['npz'])
                else:
                    raise ValueError(f"Unknown split format: {splitFormat}")
                yield featureJob, splitJob, feature_graph_name, feature_graph_files
        elif feature_name:  # For feature which does not have files
            for splitJob in utils.signac_tools.split_iter(featureJob):
                if len(splitJob.sp) == 1 and splitJob.sp.get("split_config"):
                    split_config = splitJob.sp.split_config
                else:
                    split_config = splitJob.get_id()
                feature_graph_name = "{}-{}-{}".format(
                    job.sp.graphName, feature_name, split_config)

                splitFormat = splitJob.sp.get("format", "planetoid")
                if splitFormat == "planetoid":
                    feature_graph_files = (f"{feature_graph_name}.{ext}"
                                           for ext in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index'])
                elif splitFormat == "sparsegraph":
                    feature_graph_files = (
                        f"{feature_graph_name}.{ext}" for ext in ['npz'])
                else:
                    raise ValueError(f"Unknown split format: {splitFormat}")
                yield featureJob, splitJob, feature_graph_name, feature_graph_files


@FlowProject.label
def split_generated(job: signac.Project.Job):
    split_config_exist = False
    for _, splitJob, _, feature_graph_files in feature_split_iter(job):
        split_config_exist = True
        if not splitJob.doc.get("succeeded", False):
            return False
        if not all(map(splitJob.isfile, feature_graph_files)):
            return False
    return split_config_exist


@FlowProject.operation
@FlowProject.pre(lambda job: len(list(utils.signac_tools.recursive_iter('split', graphJob=job))) > 0)
@FlowProject.pre.after(generate_feature)
@FlowProject.post(split_generated)
def generate_split(job: signac.Project.Job):
    graph = pickle.load(
        open(job.fn(job.sp.graphName + ".graph"), "rb"), encoding="bytes")
    ally = pickle.load(
        open(job.fn(job.sp.graphName + ".ally"), "rb"), encoding="bytes")
    G = nx.read_gpickle(
        job.fn(job.sp.graphName + ".gpickle.gz"))  # type: nx.Graph
    for featureJob, splitJob, feature_graph_name, feature_graph_files in feature_split_iter(job):
        if splitJob.sp.get("split_index", None) is None:
            feature_generation.random_state = reset_random_state(
                job, (job.get_id(), feature_graph_name))
        else:
            feature_generation.random_state = reset_random_state(
                job, (splitJob.get_id(), feature_graph_name))
        if all(map(splitJob.isfile, feature_graph_files)):
            print("[generate_split@{}] Skipping {}".format(
                job.get_id(), feature_graph_name))
        else:
            print("[generate_split@{}] Generating split for {}".format(
                job.get_id(), feature_graph_name))
            if featureJob.doc.get("feature_file"):
                feature_file = featureJob.fn(featureJob.doc["feature_file"])
                if splitJob.sp.get("split_source"):
                    if splitJob.sp.source_format == "GeomGCN":
                        with splitJob:
                            with np.load(splitJob.sp.split_source) as splits_file:
                                train_mask = splits_file['train_mask']
                                val_mask = splits_file['val_mask']
                                test_mask = splits_file['test_mask']
                        train_indices = np.where(train_mask)[0]
                        val_indices = np.where(val_mask)[0]
                        test_indices = np.where(test_mask)[0]
                        feature_generation.generate_split(
                            job, graph, ally, G, feature_file,
                            splitJob, feature_graph_name, feature_graph_files,
                            train_indices=train_indices,
                            validation_indices=val_indices,
                            test_indices=test_indices)
                elif job.sp.method == "planetoid":
                    # Need to merge the code which copys the files (elif -> if)
                    raise NotImplementedError()
                else:
                    feature_generation.generate_split(
                        job, graph, ally, G, feature_file, splitJob, feature_graph_name, feature_graph_files)

            elif featureJob.sp.feature_type == "unmodified":
                if job.sp.method == "planetoid":
                    for source_file, dest_file in [
                        (job.fn(f"data_source/{job.sp.datasetName}.{ext}"),
                         splitJob.fn(f"{feature_graph_name}.{ext}"))
                        for ext in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index']
                    ]:
                        shutil.copy2(source_file, dest_file)
                    assert all(map(splitJob.isfile, feature_graph_files))
                    splitJob.doc["succeeded"] = True
                    splitJob.doc["split_name"] = feature_graph_name
                else:
                    raise NotImplementedError()

            elif featureJob.sp.feature_type == "sample" and featureJob.sp.sample_type == "ogbn":
                feature_generation.ogbn_generate_split(
                    job, splitJob, feature_graph_name, feature_graph_files)
            else:
                raise ValueError()


# @FlowProject.operation
# @FlowProject.pre(FlowProject.pre.never)
# @FlowProject.post(FlowProject.post.always)
# def clear_split(job:signac.Project.Job):
#     for _, words, feature_graph_name, _ in feature_split_iter(job):
#         if "_".join(words) in job.doc.get("split_clear", []):
#             print("Removing folder {}".format(job.fn(feature_graph_name)))
#             shutil.rmtree(job.fn(feature_graph_name))


@FlowProject.operation
@FlowProject.pre(FlowProject.pre.never)
@FlowProject.post(FlowProject.post.always)
def clear_job(job: signac.Project.Job):
    workspaceDirObj = Path(job.workspace())
    for child in workspaceDirObj.iterdir():
        if child.name not in ["signac_statepoint.json", "signac_job_document.json"]:
            if child.is_dir():
                print(f"Deleting directory {child}")
                shutil.rmtree(str(child))
            else:
                print(f"Deleting {child}")
                child.unlink()


if __name__ == "__main__":
    if "--debug" in sys.argv[1:]:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

    FlowProject().main()
