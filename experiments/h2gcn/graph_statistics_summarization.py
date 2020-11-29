import signac
from pathlib import Path
from run_graph_generation import feature_split_iter, feature_generated
from collections import OrderedDict
from io import StringIO
import networkx as nx
import numpy as np
import subprocess, hashlib, sys, shutil, multiprocessing, json, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", "-o", default="exp_results/{ids}/graph_statistics.csv")
parser.add_argument("--job_id", "-i", nargs="+", default=None)


def generate_csv(job:signac.Project.Job, args):
    textBuffer = StringIO()
    args.csv_data_dict["numClass"] = job.sp.numClass
    h = job.sp.get("h", None)
    if h is not None:
        args.csv_data_dict["h"] = "{:.2f}".format(h)
    else:
        args.csv_data_dict["h"] = job.sp.source_name
    args.csv_data_dict["numNodes"] = job.doc["numNodes"]; args.csv_data_dict["numEdges"] = job.doc["numEdges"]
    args.csv_data_dict["Graph ID"] = job.get_id(); args.csv_data_dict["method"] = job.sp.method
    args.csv_data_dict["Graph Name"] = job.sp.graphName
    args.csv_data_dict["Clustering Coefficient"] = job.doc["avgClusteringCoeff"]
    args.csv_data_dict["Min Degree"] = job.doc["min_degree"]
    args.csv_data_dict["Max Degree"] = job.doc["max_degree"]
    args.csv_data_dict["Average Degree"] = job.doc["avg_degree"]
    args.csv_data_dict["AverageSPLength"] = job.doc["avgSPLength"]
    args.csv_data_dict["numTriangles"] = job.doc["numTotalTriangles"]
    args.csv_data_dict["homoEdgeRatio"] = job.doc["homoEdgeRatio"]

    G = nx.read_gpickle(job.fn(job.sp.graphName + ".gpickle.gz")) #type: nx.Graph
    args.csv_data_dict["numComponent"] = job.doc["numComponents"]
    componentSize = np.array(list(map(lambda S: len(S.nodes), nx.connected_component_subgraphs(G))))
    args.csv_data_dict["maxComponentSize"] = np.max(componentSize)
    args.csv_data_dict["meanComponentSize"] = np.mean(componentSize)
    
    assert len(args.csv_data_dict) == len(args.csv_header_list), args.csv_data_dict.keys()
    # Write to text buffer
    textBuffer.write(",".join(map(str, args.csv_data_dict.values())) + "\n")

    # Write to the result file
    if not args.csv_file_generated:
        with open(args.output, "w") as csv_out:
            csv_out.write(",".join(args.csv_header_list) + "\n")
            csv_out.write(textBuffer.getvalue())
            args.csv_file_generated = True
    else:
        with open(args.output, "a") as csv_out:
            csv_out.write(textBuffer.getvalue())

if __name__ == "__main__":
    args = parser.parse_args()

    if args.job_id is not None:
        job_id_list_md5 = hashlib.md5(bytes(' '.join(sorted(args.job_id)), encoding='utf-8')).hexdigest()
        args.output_path = args.output_path.format(ids=job_id_list_md5)
    else:
        args.output_path = args.output_path.format(ids=".")
    args.output = args.output_path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    args.csv_header_list = ["method", "numClass", "numNodes", "numEdges", "h", 
        "Graph Name", "Min Degree", "Max Degree", "Average Degree", "homoEdgeRatio", "AverageSPLength", 
        "Clustering Coefficient", "numTriangles", "numComponent", "maxComponentSize", 
        "meanComponentSize", "Graph ID"]
    args.csv_data_dict = OrderedDict.fromkeys(args.csv_header_list)
    args.csv_file_generated = False

    project = signac.get_project()
    if args.job_id is None:
        for job in project:
            generate_csv(job, args)
    else:
        for job_id in args.job_id:
            job = project.open_job(id=job_id)
            generate_csv(job, args)
    