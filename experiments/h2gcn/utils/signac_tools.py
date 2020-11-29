import signac
from pathlib import Path

def getFeatureProject(graphJob:signac.Project.Job):
    try:
        featureProject = signac.get_project(root=graphJob.workspace(), search=False)
    except LookupError:
        featureProject = signac.init_project("SyntheticExperimentFeatures", root=graphJob.workspace(), workspace="features")
    return featureProject

def feature_iter(graphJob:signac.Project.Job, sp_filter=None, doc_filter=None, include_disabled=False):
    featureProject = getFeatureProject(graphJob)
    for featureJob in featureProject.find_jobs(sp_filter, doc_filter):
        if include_disabled or (not featureJob.doc.get("disabled", False)):
            yield featureJob

def getSplitProject(featureJob:signac.Project.Job):
    try:
        splitProject = signac.get_project(root=featureJob.workspace(), search=False)
    except LookupError:
        splitProject = signac.init_project("SyntheticExperimentSplits", root=featureJob.workspace(), workspace="splits")
    return splitProject

def split_iter(featureJob:signac.Project.Job, sp_filter=None, doc_filter=None, include_disabled=False):
    splitProject = getSplitProject(featureJob)
    for splitJob in splitProject.find_jobs(sp_filter, doc_filter):
        if include_disabled or (not splitJob.doc.get("disabled", False)):
            yield splitJob

def getModelProject(splitJob:signac.Project.Job, modelRoot:str):
    projectRoot = Path(splitJob.workspace()) / modelRoot
    modelProject = signac.get_project(root=str(projectRoot), search=False)
    return modelProject

def model_iter(splitJob:signac.Project.Job, modelRoot:str, sp_filter=None, doc_filter=None):
    modelProject = getModelProject(splitJob, modelRoot)
    for modelJob in modelProject.find_jobs(sp_filter, doc_filter):
        yield modelJob

def recursive_iter(target_level:['graph', 'feature', 'split'], graphJob=None, featureJob=None, splitJob=None):
    if graphJob:
        assert (featureJob is None) and (splitJob is None)
        if target_level == "graph":
            yield graphJob
        else:
            for featureJob in feature_iter(graphJob):
                for job in recursive_iter(target_level, featureJob=featureJob):
                    yield job
    
    elif featureJob:
        assert (graphJob is None) and (splitJob is None)
        if target_level == "feature":
            yield featureJob
        else:
            for splitJob in split_iter(featureJob):
                for job in recursive_iter(target_level, splitJob=splitJob):
                    yield job
    
    elif splitJob:
        assert (featureJob is None) and (graphJob is None)
        if target_level == "split":
            yield splitJob
        else:
            raise ValueError(f"Unknown level {target_level}")

def access_proj_job(rootProjJob:signac.Project, *pathsegs):
    if len(pathsegs) == 0:
        return rootProjJob
    else:
        iter_pathseg = iter(pathsegs)
        pathseg = next(iter_pathseg)
        if pathseg == "/":
            with rootProjJob:
                try:
                    rootProjJob = signac.get_project(root=next(iter_pathseg), search=False)
                except StopIteration:
                    return signac.get_project(root=".", search=False)
        else:
            if type(rootProjJob) == signac.Project.Job:
                rootProjJob = signac.get_project(root=rootProjJob.workspace(), search=False)
            rootProjJob = rootProjJob.open_job(id=pathseg)
        return access_proj_job(rootProjJob, *list(iter_pathseg))


