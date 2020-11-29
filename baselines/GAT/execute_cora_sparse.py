import time
import scipy.sparse as sp
import numpy as np
import tensorflow as tf
import argparse, json

from models import GAT
from models import SpGAT
from utils import process


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--nb_epochs", default=1500, type=int)
parser.add_argument("--patience", default=100, type=int)
parser.add_argument("--lr", "--learning_rate", default=0.005, type=float)
parser.add_argument("--l2_coef", default=0.0005, type=float)
parser.add_argument("--hid_units", default=[8], nargs="*", type=int)
parser.add_argument("--n_heads", default=[8, 1], nargs="*", type=int)
parser.add_argument("--residual", default=False, action="store_true")
parser.add_argument("--nonlinearity", default="tf.nn.elu", type=str)
parser.add_argument("--model", default="SpGAT", type=str)
parser.add_argument("--checkpt_file", default="pre_trained/cora/mod_cora.ckpt", type=str, dest="_checkpt_file")
parser.add_argument("--dataset", default="ind.cora", type=str)
parser.add_argument("--run_id", default="", type=str)
parser.add_argument("--nhood", default=1.0, type=float)
parser.add_argument("--use_signac", default=False, action="store_true")
parser.add_argument("--signac_root", default=None, dest="_signac_root")
parser.add_argument("--dataset_path", default="data", type=str, dest="_dataset_path")
parser.add_argument("--gpu_limit", default=0, type=float, dest="_gpu_limit")
parser.add_argument("--debug", action="store_true", dest="_debug")
parser.add_argument("--attn_drop", default=0.6, type=float)
parser.add_argument("--ffd_drop", default=0.6, type=float)
parser.add_argument("--val_size", default=500, type=int)
parser.add_argument("--identity_feature", default=False, action="store_true")
parser.add_argument("--no_feature_normalize", action="store_true")

args = parser.parse_args()
if args._debug:
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
    ptvsd.wait_for_attach()
    breakpoint()

if args.nhood != float('inf') and (not np.isnan(args.nhood)):
    args.nhood = int(args.nhood)

if args.use_signac:
    import signac
    project = signac.get_project(root=args._signac_root)
    job_dict = {name: value for name, value in vars(args).items() if not name.startswith("_")}
    job = project.open_job(job_dict).init()
    args._checkpt_file = job.fn("checkpt_file/model.ckpt")

checkpt_file = args._checkpt_file

dataset = args.dataset
# training params
batch_size = args.batch_size
nb_epochs = args.nb_epochs
patience = args.patience
lr = args.lr  # learning rate
l2_coef = args.l2_coef  # weight decay
hid_units = args.hid_units # numbers of hidden units per each attention head in each layer
n_heads = args.n_heads # additional entry for the output layer
residual = args.residual
nonlinearity = eval(args.nonlinearity)
model = eval(args.model)

record_dict = dict()
tf_config = tf.ConfigProto()
if args._gpu_limit:
    tf_config.gpu_options.per_process_gpu_memory_fraction = args._gpu_limit
    print("Limit GPU Memory usage to {:.0%}".format(args._gpu_limit))
else:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

sparse = True

dataset = process.PlanetoidData(dataset, args._dataset_path, val_size=args.val_size)
if args.identity_feature:
    dataset.set_identity_features()
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = dataset.load_data_result
if not args.no_feature_normalize:
    features, spars = process.preprocess_features(features)
else:
    features, spars = features.todense(), process.sparse_to_tuple(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

features = features[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

if sparse:
    biases = process.preprocess_adj_bias(adj)
else:
    adj = adj.todense()
    adj = adj[np.newaxis]
    biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        if sparse:
            #bias_idx = tf.placeholder(tf.int64)
            #bias_val = tf.placeholder(tf.float32)
            #bias_shape = tf.placeholder(tf.int64)
            bias_in = tf.sparse_placeholder(dtype=tf.float32)
        else:
            bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session(config=tf_config) as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = features.shape[0]

            while tr_step * batch_size < tr_size:
                if sparse:
                    bbias = biases
                else:
                    bbias = biases[tr_step*batch_size:(tr_step+1)*batch_size]

                _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                    feed_dict={
                        ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_in: bbias,
                        lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        attn_drop: 0.6, ffd_drop: 0.6})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * batch_size < vl_size:
                if sparse:
                    bbias = biases
                else:
                    bbias = biases[vl_step*batch_size:(vl_step+1)*batch_size]
                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: bbias,
                        lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                    (train_loss_avg/tr_step, train_acc_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step))

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                    record_dict["train_acc"] = train_acc_avg / tr_step
                    record_dict["train_loss"] = train_loss_avg / tr_step
                    record_dict["val_loss"] = vlss_early_model
                    record_dict["val_acc"] = vacc_early_model
                    record_dict["epoch"] = epoch
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)

        ts_size = features.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            if sparse:
                bbias = biases
            else:
                bbias = biases[ts_step*batch_size:(ts_step+1)*batch_size]
            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: bbias,
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)
        record_dict["test_loss"] = ts_loss / ts_step; record_dict["test_acc"] = ts_acc / ts_step

        sess.close()

print(record_dict)
if args.use_signac:
    with open(job.fn("results.json"), "w") as f:
        json.dump(record_dict, f)
    print("Results recorded to {}".format(job.fn("results.json")))
    job.data[f"correct_label"] = dataset.labels

for scope in ('train', 'val', 'test'):
    if args.use_signac:
        job.data[f"{scope}_mask"] = getattr(dataset, f"{scope}_mask")