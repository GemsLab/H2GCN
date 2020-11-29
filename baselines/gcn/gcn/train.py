from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import json, random
import numpy as np

from utils import *
from models import GCN, MLP, BeliefGCN, GCNConcat, GCNConcat2, GCNChebyAlt1, GCNChebyAlt2


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'ind.cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('dataset_path', 'data', 'Path to dataset')
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('run_id', '', 'Name of this training session')
flags.DEFINE_bool('use_signac', False, 'Use signac and put all args into signac workspace.')
flags.DEFINE_string('signac_root', None, 'Root path for signac project.')
flags.DEFINE_string('save_plot', None, 'Path to save the plot of input data')
flags.DEFINE_bool('debug', False, 'Debug code in VS Code')
flags.DEFINE_integer('random_seed', 123, 'Random seed used in the experiment')
flags.DEFINE_integer('val_size', 500, 'Size of validation set')
flags.DEFINE_bool('_feature_normalize', True, 'Control Feature Normalize')
flags.DEFINE_integer('eigenvalue', -1, '')

# Set random seed
seed = FLAGS.random_seed
np.random.seed(seed)
tf.set_random_seed(seed)

if FLAGS.debug:
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
    ptvsd.wait_for_attach()
    breakpoint()

if FLAGS.use_signac:
    import signac
    project = signac.get_project(root=FLAGS.signac_root)
    job = project.open_job(dict(
        dataset=FLAGS.dataset, run_id=FLAGS.run_id, model=FLAGS.model, 
        learning_rate=FLAGS.learning_rate, epochs=FLAGS.epochs, hidden1=FLAGS.hidden1,
        dropout=FLAGS.dropout, weight_decay=FLAGS.weight_decay,
        early_stopping=FLAGS.early_stopping, max_degree=FLAGS.max_degree, random_seed=FLAGS.random_seed
    )).init()
    FLAGS.save_plot = job.fn(FLAGS.dataset + "-gcn-plot.pdf")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

# Load data
dataset = PlanetoidData(FLAGS.dataset, FLAGS.dataset_path, val_size=FLAGS.val_size)
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = dataset.load_data_result

# Some preprocessing
if FLAGS._feature_normalize:
    features = preprocess_features(features)
else:
    print("GCN feature normalize disabled.")
    features = sparse_to_tuple(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    if FLAGS.eigenvalue >= 0:
        support = chebyshev_polynomials(adj, FLAGS.max_degree, FLAGS.eigenvalue)
    else:
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'gcn_concat_2':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCNConcat2
elif FLAGS.model == 'gcn_cheby_concat_2':
    if FLAGS.eigenvalue >= 0:
        support = chebyshev_polynomials(adj, FLAGS.max_degree, FLAGS.eigenvalue)
    else:
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCNConcat2
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

def predict(features, support, labels, mask, placeholder):
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    prediction_prob = sess.run(model.predict(), feed_dict=feed_dict_val)
    prediction_class = np.argmax(prediction_prob, axis=1)
    return prediction_prob, prediction_class

def calculate_confusion_matrix(prediction_class, dataset, *scope, **kwargs):
    sample_mask = dataset.get_sample_mask(slice(None), *scope)
    correct_label = dataset.labels[sample_mask]
    predicted_label = prediction_class[sample_mask]
    assert len(correct_label) == len(predicted_label)
    
    confusion_matrix = np.zeros((dataset.num_labels, dataset.num_labels))
    for i in range(len(correct_label)):
        confusion_matrix[correct_label[i], predicted_label[i]] += 1

    if kwargs.get("detail_output"):
        return confusion_matrix, correct_label
    else:
        return confusion_matrix

# Init variables
sess.run(tf.global_variables_initializer())
print("Number of parameters: ", 
    np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))


cost_val = []

record_dict = dict()
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    record_dict.update(dict(
        epoch=int(epoch + 1), train_loss=float(outs[1]), train_acc=float(outs[2]),
        val_loss=float(cost), val_acc=float(acc), time=str(time.time() - t), early_stopping=False
        ))
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        record_dict["early_stopping"] = True
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
record_dict.update(test_loss=float(test_cost), test_accuracy=float(test_acc), test_duration=str(test_duration))

predicted_prob, predicted_class = predict(features, support, y_test, test_mask, placeholders)

if FLAGS.use_signac:
    with open(job.fn("results.json"), "w") as f:
        json.dump(record_dict, f)
    job.data[f"predicted_prob"] = predicted_prob
    job.data[f"correct_label"] = dataset.labels

for scope, y_scope, scope_mask in (
    ('train', y_train, train_mask),
    ('val', y_val, val_mask),
    ('test', y_test, test_mask),
    ('wild', dataset.y_wild, dataset.wild_mask)
):
    # confusion_matrix = calculate_confusion_matrix(predicted_class, dataset, scope)
    # print(f"Confusion matrix on {scope} set: ([i,j] -> node of class i classified as class j)")
    # print(confusion_matrix)
    
    if FLAGS.use_signac:
        # job.data[f"confusion_matrix_{scope}"] = confusion_matrix
        job.data[f"{scope}_mask"] = scope_mask