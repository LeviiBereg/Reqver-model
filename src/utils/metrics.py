import tensorflow as tf


def cos_loss(dummy, pred_score):
    loss_margin = 0.5
    eps = 1e-6
    neg_matrix = tf.linalg.diag(-tf.linalg.diag_part(pred_score))
    per_sample_loss = tf.maximum(eps, 
                                 loss_margin
                                    - tf.linalg.diag_part(pred_score)
                                    + tf.reduce_mean(tf.nn.relu(pred_score+neg_matrix),axis=-1))
    loss = tf.reduce_mean(per_sample_loss)
    return loss

def mrr(dummy, pred_score):
    correct_scores = tf.linalg.diag_part(pred_score)
    compared_scores = pred_score >= tf.expand_dims(correct_scores, axis=-1)
    mrr = 1 / tf.reduce_sum(tf.cast(compared_scores, dtype=tf.float32), axis=1)
    return mrr

def frank(dummy, pred_score):
    correct_scores = tf.linalg.diag_part(pred_score)
    retrieved_before = pred_score > tf.expand_dims(correct_scores, axis=-1)
    rel_ranks = tf.reduce_sum(tf.cast(retrieved_before, dtype=tf.float32), axis=1) + 1
    return rel_ranks

def relevantatk(pred_score, k):
    correct_scores = tf.linalg.diag_part(pred_score)
    compared_scores = pred_score > tf.expand_dims(correct_scores, axis=-1)
    compared_scores = tf.reduce_sum(tf.cast(compared_scores, dtype=tf.float32), axis=1)
    compared_scores = tf.cast(compared_scores < k, dtype=tf.float32)
    return compared_scores

def relevantat10(dummy, pred_score):
    return relevantatk(pred_score, k=10)

def relevantat5(dummy, pred_score):
    return relevantatk(pred_score, k=5)

def relevantat1(dummy, pred_score):
    return relevantatk(pred_score, k=1)