import experiments_workflow as workflow
workflow.model_path = "../../baselines/GAT"
workflow.workspaceRoot = "GAT_experiments"
workflow.expCode = "GAT_exp"
workflow.expProjectName = "GATSyntheticExperiments"
workflow.modelScript = "execute_cora_sparse.py"

def extra_arguments_func(task_parser):
    task_parser.add_argument("--gpu_limit", type=float, default=0)

def dataset_args_func(dataset_dir, feature_graph_name, run_id, workspaceDirObj, task_args, splitJob, **kwargs):
    argList = [
        "--dataset_path", dataset_dir, "--dataset", feature_graph_name,
        f'--run_id={run_id}', "--use_signac", "--signac_root", str(workspaceDirObj),
        "--gpu_limit", str(task_args.gpu_limit)
    ]
    if splitJob.doc.get("val_size"):
        argList += ["--val_size", str(splitJob.doc.val_size)]
    return argList

workflow.extra_arguments_func = extra_arguments_func
workflow.dataset_args_func = dataset_args_func

workflow.main()
