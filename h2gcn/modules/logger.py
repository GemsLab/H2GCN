from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import shutil
import tempfile
import numpy as np
import json


def add_subparser_args(parser):
    subparser = parser.add_argument_group(
        "Logging arguments (modules/logger.py)")
    subparser.add_argument("--checkpoint_dir", type=str,
                           default="results/checkpoints/{model}_{dataset}_{runname}")
    subparser.add_argument("--checkpoint_name", type=str,
                           default="{model}_{dataset}_{{epoch:04d}}_ta{{test_accuracy:.4f}}_va{{val_acc:.4f}}")
    subparser.add_argument("--message", "-m", default=None,
                           help="Comments appended after runname")
    subparser.add_argument("--run_id",
                           default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                           help="(default: %(default)s)")
    parser.function_hooks["argparse"].append(init_checkpoint_path)


def init_checkpoint_path(args):
    if not args.use_signac:
        if args.message is not None:
            args.run_id = args.run_id + "-" + args.message
        args.objects["checkpoint_dir"] = args.checkpoint_dir.format(
            runname=args.run_id, model=args.model, dataset=args.dataset)
    else:
        args.objects["checkpoint_dir"] = str(
            Path(args.objects["signac_job"].workspace()) / "checkpoints")
    args.objects["checkpoint_name"] = args.checkpoint_name.format(
        model=args.model, dataset=args.dataset)
    if Path(args.objects["checkpoint_dir"]).exists():
        mvTarget = tempfile.mkdtemp(prefix="checkpoints_", dir=Path(
            args.objects["checkpoint_dir"]).parent)
        Path(args.objects["checkpoint_dir"]).replace(mvTarget)
    Path(args.objects["checkpoint_dir"]).mkdir(
        parents=True)  # pylint: disable=no-member
    print("===> Checkpoint will be saved to {}".format(
        args.objects["checkpoint_dir"]))


def pretrain_logger(args, model: keras.models.Model, train_sequence, test_sequence):
    train_metrics = dict(
        zip(model.metrics_names, model.evaluate(train_sequence)))
    val_metrics = dict(zip(
        ("val_" + name for name in model.metrics_names), model.evaluate(test_sequence)))
    checkpoint_path = args.checkpoint_dir.format(
        epoch=0, **train_metrics, **val_metrics)
    print("Pretrain: saving model to {}".format(checkpoint_path))
    keras.models.save_model(model, checkpoint_path)


def save_ckpt(checkpoint, args, epoch, epoch_stats):
    ckpt_name = args.objects["checkpoint_name"].format(
        epoch=epoch, **epoch_stats)
    ckpt_path = str(Path(args.objects["checkpoint_dir"]) / ckpt_name / "ckpt")
    checkpoint.save(file_prefix=ckpt_path)
    # print(ckpt_path)
    return ckpt_name


def remove_ckpt(args, ckpt_name):
    if ckpt_name is None:
        return
    ckpt_path = Path(args.objects["checkpoint_dir"]) / ckpt_name
    if ckpt_path.exists():
        # print(f"Removing {ckpt_path}")
        shutil.rmtree(str(ckpt_path))


def restore_ckpt(checkpoint, args, ckpt_name):
    ckpt_folder = Path(args.objects["checkpoint_dir"]) / ckpt_name
    ckpt_path = tf.train.latest_checkpoint(ckpt_folder)
    return checkpoint.restore(ckpt_path)


class EpochStatsPrinter:
    def __init__(self, format_str=None):
        if format_str:
            self.format_str = format_str
        else:
            self.format_str = "    ".join([
                "Epoch: {epoch:04}", "Train Loss: {train_loss:9.6f}", "Train Acc: {train_acc:7.2%}",
                "Val Loss: {val_loss:9.6f}", "Val Acc: {val_acc:7.2%}", "Test Acc: {test_accuracy:7.2%}"
            ])

    def __call__(self, epoch, epoch_stats: dict):
        print(self.format_str.format(epoch=epoch, **epoch_stats))

    def from_dict(self, epoch_stats: dict):
        print(self.format_str.format(**epoch_stats))
        if "monitor" in epoch_stats:
            print(epoch_stats["monitor"])
