import signac
from flow import FlowProject
from pathlib import Path
from run_graph_generation import feature_split_iter, feature_generated
import subprocess
import hashlib
import sys
import shutil
import re
import argparse
import json
import utils
import logging
import tempfile
import time


model_path = None
modelScript = None
expCode = None
workspaceRoot = None
expProjectName = None
condaEnv = None

# Function references


def extra_arguments_func(task_parser): return None


dataset_args_func = lambda **kwargs: None

# Global switches
task_args = None
other_args = None
modelPathObj = None
flags = argparse.Namespace()
flags.log_to_terminal = False


def main():
    global task_args, other_args, modelPathObj
    assert all(map(lambda x: x is not None, [
        model_path, modelScript, expCode, workspaceRoot, expProjectName
    ]))
    modelPathObj = Path(__file__).parent / model_path

    task_parser = argparse.ArgumentParser(add_help=False)
    task_parser.add_argument("--config", "-c", default=None)
    task_parser.add_argument("--exp_regex", default=None)
    task_parser.add_argument("--arg_regex", default=None)
    task_parser.add_argument("--model_args", default="", type=str)
    task_parser.add_argument("--clear_workspace", action="store_true")
    task_parser.add_argument("--clean_workspace", action="store_true")

    if ("run" in sys.argv[1:]) or ("exec" in sys.argv[1:]) or ("--help" in sys.argv[1:]):
        task_parser.add_argument("--tuning", action="store_true")
        task_parser.add_argument("--use_env", default=None)
        task_parser.add_argument("--interactive", "-i", action="store_true")
        task_parser.add_argument("--exp_tags", nargs="+", default=[])
        task_parser.add_argument("--check_paths", action="store_true")


        extra_arguments_func(task_parser)
    task_args, other_args = task_parser.parse_known_args()
    if not hasattr(task_args, "model_args"):
        task_args.model_args = ""

    task_args.split_filter = None
    task_args.split_doc_filter = None

    if task_args.config:
        with open(task_args.config, "r") as args_in:
            argsDict = json.load(args_in)
            if "model_args" in argsDict:
                if task_args.model_args:
                    task_args.model_args = [
                        (arg + " " + task_args.model_args) for arg in argsDict["model_args"]]
                else:
                    task_args.model_args = argsDict["model_args"]
            if "exp_regex" in argsDict:
                task_args.exp_regex = argsDict["exp_regex"]
            if "arg_regex" in argsDict:
                task_args.arg_regex = argsDict["arg_regex"]
            if "exp_tags" in argsDict:
                task_args.exp_tags = argsDict["exp_tags"]
            if "split_filter" in argsDict:
                task_args.split_filter = argsDict["split_filter"]
            if "split_doc_filter" in argsDict:
                task_args.split_doc_filter = argsDict["split_doc_filter"]
            if "-f" not in other_args:
                if "graph_filter_dict" in argsDict:
                    graph_filter_dict = argsDict["graph_filter_dict"]
                    sys.argv += ["-f", json.dumps(graph_filter_dict)]
                    print(sys.argv)
                elif "graph_filter" in argsDict:
                    graph_filter = argsDict["graph_filter"]
                    sys.argv += ["-f"] + graph_filter
                    print(sys.argv)
                
    if task_args.clean_workspace:
        sys.argv += ["-o", "clean_workspace"]
    if task_args.clear_workspace:
        sys.argv += ["-o", "clear_workspace"]

    if task_args.model_args and type(task_args.model_args) is str:
        task_args.model_args = task_args.model_args.split(";")
        print("Model args: {}".format(task_args.model_args))

    if not (modelPathObj.exists() and modelPathObj.is_dir()):
        raise ValueError("Path {} does not exist or is not a folder. \n"
                         "Please change the 'model_path' variable in the script.".format(model_path))

    if ("--debug" in other_args) or ("--show-traceback" in other_args) or vars(task_args).get("tuning", False):
        flags.log_to_terminal = True
        print("Log to terminal enabled.")

    if ("status" in sys.argv[1:]):
        while sys.argv[1] != "status":
            del sys.argv[1]
        FlowProject().main()
    else:
        task_help_parser = argparse.ArgumentParser(parents=[task_parser])
        FlowProject().main(parser=task_help_parser)

    if task_args.model_args:
        print("Model args: {}".format(task_args.model_args))


def calculate_md5(file_path):
    return hashlib.md5(open(file_path, "rb").read()).hexdigest()


def model_experiments_needed(job: signac.Project.Job):
    if task_args.check_paths:
        return False
    elif is_tuning():
        return True
    elif task_args.model_args:
        return True
    else:
        return any([len(splitJob.doc.get(expCode, [])) > 0
                    for splitJob in utils.signac_tools.recursive_iter('split', graphJob=job)])


def get_exp_regex(job):
    if hasattr(task_args, "exp_regex"):
        return (task_args.exp_regex or job.doc.get("exp_regex", default=""))
    else:
        return job.doc.get("exp_regex", default="")


def is_tuning():
    return vars(task_args).get("tuning", False)


condaEnvDict = dict()


def get_python_path():
    global condaEnv
    use_current_env = False
    if task_args.use_env == ".":
        use_current_env = True
    elif task_args.use_env is not None:
        condaEnv = task_args.use_env
    if condaEnv and not use_current_env:
        if condaEnvDict.get(condaEnv) is None:
            pythonPath = subprocess.run(["conda", "run", "-n", condaEnv, "which", "python"],
                                        stdout=subprocess.PIPE, encoding="utf8", check=True).stdout.strip()
            condaEnvDict[condaEnv] = pythonPath
        else:
            pythonPath = condaEnvDict[condaEnv]
        print(f"Using {pythonPath}")
    else:
        pythonPath = sys.executable
    return pythonPath


@FlowProject.label
def model_experiments_finished(job: signac.Project.Job, key="succeeded"):
    if not model_experiments_needed(job):
        return False

    for featureJob, splitJob, feature_graph_name, feature_graph_files in feature_split_iter(job):
        if re.search(get_exp_regex(job), feature_graph_name) is None:
            continue
        elif splitJob not in utils.signac_tools.getSplitProject(featureJob).find_jobs(
                task_args.split_filter, task_args.split_doc_filter):
            continue
        md5_str = "_".join(map(lambda x: calculate_md5(
            splitJob.fn(x)), feature_graph_files))

        exp_args_list = task_args.model_args or splitJob.doc.get(
            expCode, default=[])
        if exp_args_list == [] and is_tuning():
            exp_args_list = [""]
        for args in exp_args_list:
            if task_args.arg_regex is not None and re.search(task_args.arg_regex, args) is None:
                continue
            dataset_dir = splitJob.workspace()
            datasetDirObj = Path(dataset_dir)

            # Workspace path
            workspaceDirObj = datasetDirObj / workspaceRoot  # type:Path
            if not (workspaceDirObj.exists() and workspaceDirObj.is_dir()):
                return False
            modelProject = signac.init_project(
                name=expProjectName, root=str(workspaceDirObj))
            run_id = "{}@{}".format(args, md5_str)
            if is_tuning():
                run_id += "[tuning]"
            if not any(map(lambda job_i: job_i.doc.get(key, False), modelProject.find_jobs(filter={"run_id": run_id}))):
                return False
    return True


@FlowProject.operation
@FlowProject.pre(model_experiments_needed)
@FlowProject.pre(feature_generated)
@FlowProject.post(model_experiments_finished)
def run_model(job: signac.Project.Job):
    logger = logging.getLogger('run_model@{}'.format(job.get_id()))
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    chFormatter = logging.Formatter(
        '[{asctime} {name} {levelname:>8}] {message}', '%m-%d %H:%M:%S', '{')
    ch.setFormatter(chFormatter)
    logger.addHandler(ch)

    for featureJob, splitJob, feature_graph_name, feature_graph_files in feature_split_iter(job):
        exp_regex = get_exp_regex(job)
        if re.search(exp_regex, feature_graph_name) is None:
            print("[run_model@{}] Regex {} not matching; skip on dataset {}".format(
                job.get_id(), exp_regex, feature_graph_name))
            continue
        elif splitJob not in utils.signac_tools.getSplitProject(featureJob).find_jobs(
                task_args.split_filter, task_args.split_doc_filter):
            print("[run_model@{}] Filter {} not matching; skip on dataset {}".format(
                job.get_id(),
                (task_args.split_filter, task_args.split_doc_filter),
                feature_graph_name))
            continue
        elif is_tuning() and (splitJob.sp.get("split_index", None) not in {None, 0}):
            print("[run_model@{}] Split index is not 0 for tuning; skip on dataset {}".format(
                job.get_id(), feature_graph_name))
            continue
        md5_str = "_".join(map(lambda x: calculate_md5(
            splitJob.fn(x)), feature_graph_files))
        dataset_dir = splitJob.workspace()
        datasetDirObj = Path(dataset_dir)

        # Workspace path
        workspaceDirObj = datasetDirObj / workspaceRoot
        workspaceDirObj.mkdir(exist_ok=True, parents=True)
        modelProject = signac.init_project(
            name=expProjectName, root=str(workspaceDirObj))

        fh = logging.FileHandler(
            str(workspaceDirObj / "terminal_output.log"), "a")
        fh.setLevel(logging.DEBUG)
        fhFormatter = logging.Formatter(
            '[{asctime} {levelname:>8}] {message}', '%m-%d %H:%M:%S', '{')
        fh.setFormatter(fhFormatter)
        logger.addHandler(fh)

        exp_args_list = task_args.model_args or splitJob.doc.get(
            expCode, default=[])
        if exp_args_list == [] and is_tuning():
            exp_args_list = [""]
        for args in exp_args_list:
            if task_args.arg_regex is not None and re.search(task_args.arg_regex, args) is None:
                print("[run_model@{}] Regex {} not matching; skip on args {}".format(
                    job.get_id(), task_args.arg_regex, args))
                continue
            run_id = "{}@{}".format(args, md5_str)
            if is_tuning():
                run_id += "[tuning]"
                logger.removeHandler(fh)
            if any(map(lambda job_i: job_i.doc.get("succeeded", False), modelProject.find_jobs(filter={"run_id": run_id}))):
                print("[run_model@{}] Already run; skip on dataset {} for parameter {}".format(
                    job.get_id(), feature_graph_name, args))
            else:
                # Construct arguments
                args_split = args.split()
                dataset_args = dataset_args_func(
                    dataset_dir=dataset_dir, feature_graph_name=feature_graph_name,
                    run_id=run_id, workspaceDirObj=workspaceDirObj, task_args=task_args,
                    featureJob=featureJob, args=args, args_split=args_split, splitJob=splitJob
                )
                if dataset_args is None:
                    raise ValueError(
                        "dataset_args_func is not properly configured.")
                elif dataset_args is False:
                    print("[run_model@{}] Skip on dataset {} for parameter {}".format(
                        job.get_id(), feature_graph_name, args))
                    continue
                arg_list = [get_python_path(), "-u", modelScript] + \
                    dataset_args + args_split

                # Run model code
                print("[run_model@{}] run on dataset {} for parameter {}".format(
                    job.get_id(), feature_graph_name, args))
                try:
                    logger.info(
                        "===============\n>>>>Executing command {}\n===============".format(arg_list))
                    if not(job.doc.get("exp_terminal", False) or flags.log_to_terminal):
                        ch.setLevel(logging.WARNING)
                        ch.setFormatter(chFormatter)
                    if task_args.interactive:
                        proc = subprocess.Popen(
                            arg_list, cwd=str(modelPathObj))
                    else:
                        proc = subprocess.Popen(arg_list, cwd=str(modelPathObj),
                                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
                    if proc.stdout is not None:
                        msgcount = 0
                        for line in iter(proc.stdout.readline, ''):
                            msgcount += 1
                            logger.info(line.strip())
                            if msgcount % 100 == 0:
                                logger.debug("running on dataset {} for parameter {}".format(
                                    feature_graph_name, args))
                                msgcount = 0
                    returncode = proc.wait()
                    if returncode != 0:
                        raise subprocess.CalledProcessError(
                            returncode, arg_list)
                    else:
                        logger.debug("Completed on dataset {} for parameter {}".format(
                            feature_graph_name, args))
                except subprocess.CalledProcessError:
                    logger.error("Check log at {}".format(
                        workspaceDirObj / "terminal_output.log"))
                    raise
                logger.info("===============")
                ch.setLevel(logging.INFO)

                # Tag job as succeeded (except when tuning)
                assert len(modelProject.find_jobs(
                    filter={"run_id": run_id})) == 1
                if not task_args.tuning:
                    for job_m in modelProject.find_jobs(filter={"run_id": run_id}):
                        job_m.doc["succeeded"] = True
                else:
                    print("[run_model@{}]Job will not be tagged successful in tuning mode.".format(
                        job.get_id()))
        logger.removeHandler(fh)


@FlowProject.operation
@FlowProject.pre(lambda job: task_args.check_paths)
@FlowProject.post(lambda job: False)
def get_split_paths(job: signac.Project):
    for featureJob, splitJob, feature_graph_name, feature_graph_files in feature_split_iter(job):
        exp_regex = get_exp_regex(job)
        if re.search(exp_regex, feature_graph_name) is None:
            continue
        elif splitJob not in utils.signac_tools.getSplitProject(featureJob).find_jobs(
                task_args.split_filter, task_args.split_doc_filter):
            continue
        elif is_tuning() and (splitJob.sp.get("split_index", None) not in {None, 0}):
            continue
        dataset_dir = Path(splitJob.workspace())
        print(dataset_dir / feature_graph_name)
    

@FlowProject.operation
@FlowProject.pre(lambda job: task_args.clear_workspace)
@FlowProject.post(lambda job: False)
def clear_workspace(job: signac.Project.Job):
    for featureJob, splitJob, feature_graph_name, feature_graph_files in feature_split_iter(job):
        exp_regex = get_exp_regex(job)
        if re.search(exp_regex, feature_graph_name) is None:
            print("[run_model@{}] Regex {} not matching; skip on dataset {}".format(
                job.get_id(), exp_regex, feature_graph_name))
            continue
        elif splitJob not in utils.signac_tools.getSplitProject(featureJob).find_jobs(
                task_args.split_filter, task_args.split_doc_filter):
            print("[run_model@{}] Filter {} not matching; skip on dataset {}".format(
                job.get_id(),
                (task_args.split_filter, task_args.split_doc_filter),
                feature_graph_name))
            continue
        dataset_dir = splitJob.workspace()
        datasetDirObj = Path(dataset_dir)
        # Workspace path
        workspaceDirObj = datasetDirObj / workspaceRoot  # type: Path
        if workspaceDirObj.exists():
            assert workspaceDirObj.is_dir()
            if task_args.model_args:
                try:
                    modelProject = signac.get_project(
                        root=str(workspaceDirObj), search=False)
                    md5_str = "_".join(map(lambda x: calculate_md5(
                        splitJob.fn(x)), feature_graph_files))
                    for args in task_args.model_args:
                        if task_args.arg_regex is not None and re.search(task_args.arg_regex, args) is None:
                            print("[run_model@{}] Regex {} not matching; skip on args {}".format(
                                job.get_id(), task_args.arg_regex, args))
                            continue
                        run_id = "{}@{}".format(args, md5_str)
                        for model_job in modelProject.find_jobs(filter={"run_id": run_id}):
                            print("Removing folder {}".format(
                                model_job.workspace()))
                            shutil.rmtree(model_job.workspace())
                except LookupError:
                    pass
            else:
                print("Removing folder {}".format(workspaceDirObj))
                shutil.rmtree(str(workspaceDirObj))


@FlowProject.operation
@FlowProject.pre(lambda job: task_args.clean_workspace)
@FlowProject.post(lambda job: False)
def clean_workspace(job: signac.Project.Job):
    for _, splitJob, feature_graph_name, feature_graph_files in feature_split_iter(job):
        exp_regex = get_exp_regex(job)
        if re.search(exp_regex, feature_graph_name) is None:
            print("[run_model@{}] Regex {} not matching; skip on dataset {}".format(
                job.get_id(), exp_regex, feature_graph_name))
            continue
        dataset_dir = splitJob.workspace()
        datasetDirObj = Path(dataset_dir)
        if all(map(splitJob.isfile, feature_graph_files)):
            md5_str = "_".join(map(lambda x: calculate_md5(
                splitJob.fn(x)), feature_graph_files))
        else:
            md5_str = None
            print(
                f"[clean_workspace@{job.get_id()}] Missing files for split {feature_graph_name}")

        # Workspace path
        workspaceDirObj = datasetDirObj / workspaceRoot  # type: Path
        if workspaceDirObj.exists():
            assert workspaceDirObj.is_dir()
            try:
                modelProject = signac.get_project(
                    root=str(workspaceDirObj), search=False)
                for model_job in modelProject:
                    if not model_job.doc.get("succeeded", False):
                        target_dir = model_job.workspace()
                        print(
                            f"[clean_workspace@{job.get_id()}] Experiment not succeeded: removing folder {target_dir}")
                        shutil.rmtree(target_dir)
                    elif (md5_str is not None) and (not model_job.sp.run_id.endswith(md5_str)):
                        target_dir = model_job.workspace()
                        print(
                            f"[clean_workspace@{job.get_id()}] Experiment not matching current data: removing folder {target_dir}")
                        shutil.rmtree(target_dir)
            except LookupError:
                pass

