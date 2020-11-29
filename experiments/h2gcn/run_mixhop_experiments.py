import signac
from flow import FlowProject
from pathlib import Path
from run_graph_generation import feature_split_iter, feature_generated
import subprocess, hashlib, sys, shutil, re

import experiments_workflow as workflow
workflow.model_path = "../../baselines/mixhop"
workflow.workspaceRoot = "mixhop_experiments"
workflow.expCode = "mixhop_exp"
workflow.expProjectName = "mixhopSyntheticExperiments"
workflow.modelScript = "mixhop_trainer.py"

def extra_arguments_func(task_parser):
    pass

def dataset_args_func(dataset_dir, feature_graph_name, run_id, workspaceDirObj, splitJob, **kwargs):
    argList = [
        "--dataset_dir", dataset_dir, "--dataset_name", feature_graph_name,
        "--run_id", run_id, "--use_signac", "--signac_root", str(workspaceDirObj)
    ]
    if splitJob.doc.get("val_size"):
        argList += ["--num_validate_nodes", str(splitJob.doc.val_size)]
    return argList

workflow.extra_arguments_func = extra_arguments_func
workflow.dataset_args_func = dataset_args_func

workflow.main()
