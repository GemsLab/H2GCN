import signac
from pathlib import Path
from run_graph_generation import feature_split_iter, feature_generated
from collections import OrderedDict
from io import StringIO
import subprocess
import hashlib
import sys
import shutil
import multiprocessing
import json
import argparse
import hashlib
import csv
import sys

parser = argparse.ArgumentParser()
parser.add_argument("experiment_types", nargs="+")
parser.add_argument("--output_path", "-o",
                    default="exp_results/{ids}/{{experiment_type}}_results.csv")
parser.add_argument("--job_id", "-i", nargs="+", default=None)
parser.add_argument("--exp_args", "-a", nargs=argparse.REMAINDER, default=None)
parser.add_argument("--add_args", nargs="+", default=[])
parser.add_argument("--from_json", "-f", nargs="+", default=[])
parser.add_argument("--path_only", "-p", action="store_true")


def calculate_md5(file_path):
    return hashlib.md5(open(file_path, "rb").read()).hexdigest()


def mixhop_result_parser(job_m: signac.Project.Job, args):
    resultDirObj = Path(job_m.fn("results"))
    result_json_name = None
    for item in resultDirObj.iterdir():
        if item.match("*.json"):
            assert result_json_name is None
            result_json_name = str(item)
    with open(result_json_name, "r") as json_in:
        exp_metrics = json.load(json_in)
    args.csv_data_dict["Train Epoch"] = exp_metrics["at_best_validate"][2]
    args.csv_data_dict["Test Acc"] = exp_metrics["at_best_validate"][1]
    args.csv_data_dict["Val Acc"] = exp_metrics["at_best_validate"][0]


def gcn_result_parser(job_m: signac.Project.Job, args):
    with open(job_m.fn("results.json"), "r") as json_in:
        exp_metrics = json.load(json_in)
    args.csv_data_dict["Train Epoch"] = exp_metrics["epoch"]
    args.csv_data_dict["Test Acc"] = exp_metrics["test_accuracy"]
    args.csv_data_dict["Val Acc"] = exp_metrics["val_acc"]
    args.csv_data_dict["Validation Loss"] = exp_metrics["val_loss"]
    args.csv_data_dict["Train Acc"] = exp_metrics["train_acc"]
    args.csv_data_dict["Train Loss"] = exp_metrics["train_loss"]
    args.csv_data_dict["Early Stopping"] = exp_metrics["early_stopping"]


def GAT_result_parser(job_m: signac.Project.Job, args):
    with open(job_m.fn("results.json"), "r") as json_in:
        exp_metrics = json.load(json_in)
    args.csv_data_dict["Train Epoch"] = exp_metrics["epoch"]
    args.csv_data_dict["Test Loss"] = exp_metrics["test_loss"]
    args.csv_data_dict["Test Acc"] = exp_metrics["test_acc"]
    args.csv_data_dict["Val Acc"] = exp_metrics["val_acc"]
    args.csv_data_dict["Train Acc"] = exp_metrics["train_acc"]
    args.csv_data_dict["Train Loss"] = exp_metrics["train_loss"]
    args.csv_data_dict["Validation Loss"] = exp_metrics["val_loss"]


def HGCN_result_parser(job_m: signac.Project.Job, args):
    with open(job_m.fn("results.json"), "r") as json_in:
        exp_metrics = json.load(json_in)
    args.csv_data_dict["Train Epoch"] = exp_metrics["epoch"]
    args.csv_data_dict["Test Acc"] = exp_metrics["test_accuracy"]
    args.csv_data_dict["Test Loss"] = exp_metrics["test_loss"]
    args.csv_data_dict["Val Acc"] = exp_metrics["val_acc"]
    args.csv_data_dict["Validation Loss"] = exp_metrics["val_loss"]
    args.csv_data_dict["Train Acc"] = exp_metrics["train_acc"]
    args.csv_data_dict["Train Loss"] = exp_metrics["train_loss"]


def graphsage_result_parser(job_m: signac.Project.Job, args):
    with open(job_m.fn("results.json"), "r") as json_in:
        exp_metrics = json.load(json_in)
    args.csv_data_dict["Train Epoch"] = exp_metrics["epoch"]
    args.csv_data_dict["Test Acc"] = exp_metrics["test_accuracy"]
    args.csv_data_dict["Val Acc"] = exp_metrics["val_acc"]
    args.csv_data_dict["Train Acc"] = exp_metrics["train_acc"]
    args.csv_data_dict["Train Loss"] = exp_metrics["train_loss"]


def path_parser(job_m: signac.Project.Job, args):
    args.csv_data_dict["Exp Type"] = args.exp_type


def generate_csv(job: signac.Project.Job, args):
    textBuffer = StringIO()
    textList = []
    args.csv_data_dict["numClass"] = job.sp.numClass
    try:
        args.csv_data_dict["h"] = "{:.2f}".format(job.sp.h)
    except AttributeError:
        args.csv_data_dict["h"] = job.sp.HName
    args.csv_data_dict["Graph ID"] = job.get_id()
    args.csv_data_dict["Clustering Coefficient"] = job.doc.get(
        "avgClusteringCoeff")
    for featureJob, splitJob, feature_graph_name, feature_graph_files in feature_split_iter(job):
        feature_file = featureJob.doc.get("feature_file")
        if featureJob.doc.get("feature_name"):
            args.csv_data_dict["Feature"] = featureJob.doc["feature_name"]
        else:
            args.csv_data_dict["Feature"] = Path(
                feature_file.replace(job.sp.graphName + "-", "")).stem
        args.csv_data_dict["Graph Name"] = feature_graph_name
        args.csv_data_dict["Split Config"] = splitJob.sp.split_config
        md5_str = "_".join(map(lambda x: calculate_md5(
            splitJob.fn(x)), feature_graph_files))
        dataset_dir = splitJob.workspace()
        datasetDirObj = Path(dataset_dir)

        # Workspace path
        workspaceDirObj = datasetDirObj / args.workspaceRoot
        try:
            gcnProject = signac.get_project(
                root=str(workspaceDirObj), search=False)
        except LookupError as e:
            print(e, file=sys.stderr)
            continue

        if args.exp_args is not None:
            exp_arg_list = args.exp_args
        elif args.add_args:
            exp_arg_list = list(
                set(splitJob.doc.get(args.exp_type, default=[])) |
                set(args.add_args)
            )
        else:
            exp_arg_list = splitJob.doc.get(args.exp_type, default=[])

        for exp_args in exp_arg_list:
            args.csv_data_dict["Model Args"] = '"{}"'.format(exp_args)
            run_id = "{}@{}".format(exp_args, md5_str)
            job_iter = gcnProject.find_jobs(filter={"run_id": run_id})
            if any(map(lambda job_i: job_i.doc.get("succeeded", False), job_iter)):
                assert len(job_iter) == 1, (args.csv_data_dict, run_id)
                # Parse experiment results
                for job_m in job_iter:
                    args.csv_data_dict["Experiment ID"] = job_m.get_id()
                    args.result_parser(job_m, args)
                    if args.path_only:
                        path = [job.get_id(), featureJob.get_id(), splitJob.get_id(
                        ), "/", args.workspaceRoot, job_m.get_id()]
                        args.csv_data_dict["Job Path"] = json.dumps(path)
                assert len(args.csv_data_dict) == len(args.csv_header_list)

                # Write to text buffer
                textBuffer.write(
                    ",".join(map(str, args.csv_data_dict.values())) + "\n")
                textList.append(list(map(str, args.csv_data_dict.values())))

    if not args.path_only:
        # Write to the result file
        if not args.csv_file_generated:
            print(f"CSV will be saved to {args.output}")
            with open(args.output, "w") as csv_out:
                csv_out.write(",".join(args.csv_header_list) + "\n")
                csv_out.write(textBuffer.getvalue())
                args.csv_file_generated = True
        else:
            with open(args.output, "a") as csv_out:
                csv_out.write(textBuffer.getvalue())
    else:
        # Write to the result file
        csv_writer = csv.writer(sys.stdout)
        if not args.csv_file_generated:
            csv_writer.writerow(args.csv_header_list)
            csv_writer.writerows(textList)
            args.csv_file_generated = True
        else:
            csv_writer.writerows(textList)


if __name__ == "__main__":
    args = parser.parse_args()
    project = signac.get_project()
    for json_path in args.from_json:
        with open(json_path, "r") as json_in:
            confDict = json.load(json_in)
        args.add_args += confDict.get("model_args", [])
        if "graph_filter_dict" in confDict and args.job_id is None:
            args.job_id = list(
                (job.get_id() for job in project.find_jobs(confDict["graph_filter_dict"])))

    if args.job_id is not None:
        args.job_id = list(sorted(args.job_id))
        job_id_list_md5 = hashlib.md5(
            bytes(' '.join(args.job_id), encoding='utf-8')).hexdigest()
        args.output_path = args.output_path.format(ids=job_id_list_md5)
    else:
        args.output_path = args.output_path.format(ids=".")
    args.csv_file_generated = False

    for exp_type in args.experiment_types:
        exp_type = exp_type.lower()
        args.output = args.output_path.format(experiment_type=exp_type)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if exp_type == "gcn":
            args.exp_type = "gcn_exp"
            args.csv_header_list = ["numClass", "h", "Feature", "Graph Name", "Model Args",
                                    "Train Epoch", "Test Acc", "Val Acc", "Train Acc", "Train Loss", "Validation Loss", "Early Stopping",
                                    "Graph ID", "Split Config", "Experiment ID", "Clustering Coefficient"]
            args.workspaceRoot = "gcn_experiments"
            args.result_parser = gcn_result_parser
        elif exp_type == "mixhop":
            args.exp_type = "mixhop_exp"
            args.csv_header_list = ["numClass", "h", "Feature", "Graph Name", "Model Args",
                                    "Train Epoch", "Test Acc", "Val Acc",
                                    "Graph ID", "Split Config", "Experiment ID", "Clustering Coefficient"]
            args.workspaceRoot = "mixhop_experiments"
            args.result_parser = mixhop_result_parser
        elif exp_type == "gat":
            args.exp_type = "GAT_exp"
            args.csv_header_list = ["numClass", "h", "Feature", "Graph Name", "Model Args",
                                    "Train Epoch", "Test Acc", "Val Acc", "Train Acc", "Train Loss", "Validation Loss", "Test Loss",
                                    "Graph ID", "Split Config", "Experiment ID", "Clustering Coefficient"]
            args.workspaceRoot = "GAT_experiments"
            args.result_parser = GAT_result_parser
        elif exp_type == "h2gcn" or exp_type == "hgcn":
            args.exp_type = "hgcn_exp"
            args.csv_header_list = ["numClass", "h", "Feature", "Graph Name", "Model Args",
                                    "Train Epoch", "Test Acc", "Val Acc", "Train Acc", "Train Loss", "Validation Loss", "Test Loss",
                                    "Graph ID", "Split Config", "Experiment ID", "Clustering Coefficient"
                                    ]
            args.workspaceRoot = "experiments/hgcn_experiments"
            args.result_parser = HGCN_result_parser
        elif exp_type == "graphsage":
            args.exp_type = "graphsage_exp"
            args.workspaceRoot = "experiments/graphsage_experiments"
            args.csv_header_list = [
                "numClass", "h", "Feature", "Graph Name", "Model Args", "Train Epoch",
                "Test Acc", "Val Acc", "Train Acc", "Train Loss",
                "Graph ID", "Split Config", "Experiment ID", "Clustering Coefficient"
            ]
            args.result_parser = graphsage_result_parser
        else:
            raise ValueError(f"Unknown experiment type {exp_type}")

        if args.path_only:
            args.csv_header_list = ["numClass", "h", "Feature", "Graph Name", "Model Args",
                                    "Exp Type", "Job Path",
                                    "Graph ID", "Split Config", "Experiment ID", "Clustering Coefficient"]
            args.result_parser = path_parser

        args.csv_data_dict = OrderedDict.fromkeys(args.csv_header_list)

        if args.job_id is not None:
            for job_id in args.job_id:
                job = project.open_job(id=job_id)
                generate_csv(job, args)
        else:
            for job in project:
                generate_csv(job, args)
