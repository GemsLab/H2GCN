import re
import experiments_workflow as workflow
workflow.model_path = "../../baselines/gcn/gcn"
workflow.workspaceRoot = "gcn_experiments"
workflow.expCode = "gcn_exp"
workflow.expProjectName = "gcnSyntheticExperiments"
workflow.modelScript = "train.py"

def extra_arguments_func(task_parser):
    pass

def dataset_args_func(dataset_dir, feature_graph_name, run_id, workspaceDirObj, featureJob, args, splitJob, **kwargs):
    feature_file = featureJob.doc.get("feature_file")
    if re.search(r"--model\s+dense", args) and feature_file is not None and feature_file.endswith(".sample_pred"):
        return False
    else:
        argList = [
            "--dataset_path", dataset_dir, "--dataset", feature_graph_name,
            "--run_id", run_id, "--use_signac", "--signac_root", str(workspaceDirObj)
        ]
        if splitJob.doc.get("val_size"):
            argList += ["--val_size", str(splitJob.doc.val_size)]
        return argList

workflow.extra_arguments_func = extra_arguments_func
workflow.dataset_args_func = dataset_args_func

workflow.main()
