from . import *
from collections import deque


def create_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.function_hooks = dict()
    parser.function_hooks["argparse"] = deque()
    return parser


def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument("--use_signac", default=False, action="store_true")
    parser.add_argument("--signac_root", default=None, dest="_signac_root",
                        help="Root path of signac job for experiment.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--help", "-h", action="help")
    parser.add_argument("--exp_tags", default=[], nargs="+", dest="_exp_tags")

    args = parser.parse_args()
    args.objects = dict(function_hooks=parser.function_hooks)

    if args.use_signac:  # Signac functionality
        import signac
        project = signac.get_project(root=args._signac_root)
        args.objects["signac_project"] = project
        job_dict = {name: value for name, value in vars(args).items() if
                    (not name.startswith("_")) and (name != "objects")}
        args.objects["signac_job"] = project.open_job(job_dict).init()
        args.objects["signac_job"].doc["exp_tags"] = args._exp_tags

    # list of func(args, model, train_sequence, test_sequence)
    args.objects["pretrain_callbacks"] = deque()
    args.objects["pre_epoch_callbacks"] = deque()
    args.objects["post_epoch_callbacks"] = deque()
    args.objects["post_train_callbacks"] = deque()
    while len(parser.function_hooks["argparse"]) > 0:
        function_ptr = parser.function_hooks["argparse"].popleft()
        function_ptr(args)

    return args
