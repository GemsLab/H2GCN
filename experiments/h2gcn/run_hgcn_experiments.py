import experiments_workflow as workflow
workflow.model_path = "../../h2gcn"
workflow.workspaceRoot = "experiments/hgcn_experiments"
workflow.expCode = "hgcn_exp"
workflow.expProjectName = "hgcnSyntheticExperiments"
workflow.modelScript = "run_experiments.py"


def extra_arguments_func(task_parser):
    task_parser.add_argument("--grow_gpu_mem", action="store_true")


def dataset_args_func(dataset_dir, feature_graph_name, run_id, workspaceDirObj,
                      args_split, task_args, splitJob, **kwargs):
    model = args_split[0]
    datafmt = splitJob.sp.get("format", "planetoid")
    del args_split[0]
    argList = [
        model, datafmt,  "--dataset_path", dataset_dir, "--dataset",
        feature_graph_name, f"--run_id={run_id}", "--use_signac", "--signac_root", str(
            workspaceDirObj)
    ] 
    if task_args.grow_gpu_mem:
        argList += ["--grow_gpu_mem"]
    if task_args.exp_tags:
        argList += (["--exp_tags"] + task_args.exp_tags)
    if splitJob.doc.get("val_size"):
        argList += ["--val_size", str(splitJob.doc.val_size)]
    return argList


workflow.extra_arguments_func = extra_arguments_func
workflow.dataset_args_func = dataset_args_func

workflow.main()
