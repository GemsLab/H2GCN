import numpy as np
import tensorflow as tf
from models._metrics import masked_accuracy


def add_subparser_args(parser):
    subparser = parser.add_argument_group(
        "Monitor arguments (modules/monitor.py)")
    subparser.add_argument("--deg_acc_monitor",
                           default=[], type=float, nargs="+")
    subparser.add_argument(
        "--grad_monitor", default=False, action="store_true")


# Test steps


def deg_acc_monitor(args, degree_bins, adj, predictions, y_sample, sample_mask, sample_name, stats_dict=dict()):
    degree = tf.reduce_sum(adj, axis=1)
    prev_deg_mask = None
    deg_acc_results = []
    deg_counts = []
    for b in degree_bins:
        deg_mask = (degree <= b)
        if prev_deg_mask is not None:
            deg_mask_range = tf.logical_and(
                tf.logical_not(prev_deg_mask), deg_mask)
        else:
            deg_mask_range = deg_mask
        prev_deg_mask = deg_mask
        deg_mask_range = tf.logical_and(sample_mask, deg_mask_range)
        b_acc = masked_accuracy(predictions, y_sample, deg_mask_range)
        deg_acc_results.append(b_acc.numpy())
        deg_counts.append(tf.reduce_sum(tf.cast(deg_mask_range, tf.int32)).numpy())
    deg_mask_range = tf.logical_and(tf.logical_not(prev_deg_mask), sample_mask)
    b_acc = masked_accuracy(predictions, y_sample, deg_mask_range)
    deg_acc_results.append(b_acc.numpy())
    deg_counts.append(tf.reduce_sum(tf.cast(deg_mask_range, tf.int32)).numpy())
    print(f"[deg_acc_monitor - {degree_bins} - {deg_counts} - {sample_name} Acc] {deg_acc_results}")
    stats_dict[f"deg_acc_{sample_name}"] = dict()
    stats_dict[f"deg_acc_{sample_name}"]["bins"] = degree_bins
    stats_dict[f"deg_acc_{sample_name}"]["counts"] = [int(x) for x in deg_counts]
    stats_dict[f"deg_acc_{sample_name}"]["acc"] = [float(x) for x in deg_acc_results]

    if args.use_signac:
        job = args.objects["signac_job"]
        job.data[f"deg_acc/{sample_name}/bins"] = np.array(degree_bins)
        job.data[f"deg_acc/{sample_name}/counts"] = np.array(deg_counts)
        job.data[f"deg_acc/{sample_name}/acc"] = np.array(deg_acc_results)


def grad_monitor(model, gradients):
    gnp = [g.numpy() for g in gradients]
    print(f"Gradient range: " + "  ".join([
          f"[{v.name}] ({g.min():.2e}, {np.abs(g).min():.2e}, {g.max():.2e})" for v, g in zip(model.trainable_variables, gnp)]))
