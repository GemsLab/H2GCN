import tensorflow as tf

###############################################
# This section of code adapted from tkipf/gcn #
# Modified by Jiong Zhu (jiongzhu@umich.edu)  #
###############################################

@tf.function
def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_sum(mask)
    loss *= mask
    return tf.reduce_sum(loss)

@tf.function
def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_sum(mask)
    accuracy_all *= mask
    return tf.reduce_sum(accuracy_all)

