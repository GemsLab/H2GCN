import re
import experiments_workflow as workflow
workflow.model_path = "../../baselines/graphsage-simple/graphsage"
workflow.workspaceRoot = "experiments/graphsage_experiments"
workflow.expCode = "graphsage_exp"
workflow.expProjectName = "graphsageSyntheticExperiments"
workflow.modelScript = "model.py"


def extra_arguments_func(task_parser):
    pass


def dataset_args_func(dataset_dir, feature_graph_name, run_id, workspaceDirObj,
                      featureJob, args, splitJob, **kwargs):
    argList = [
        "--dataset_path", dataset_dir, "--dataset", feature_graph_name,
        f"--run_id={run_id}", "--use_signac", "--signac_root", str(
            workspaceDirObj)
    ]
    if splitJob.doc.get("val_size"):
        argList += ["--val_size", str(splitJob.doc.val_size)]
    return argList


workflow.extra_arguments_func = extra_arguments_func
workflow.dataset_args_func = dataset_args_func

workflow.main()
