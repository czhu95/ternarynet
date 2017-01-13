#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

G = tf.get_default_graph()

def p_ternarize(x, p):

    x = tf.tanh(x)
    shape = x.get_shape()

    thre = tf.get_variable('T', trainable=False, collections=[tf.GraphKeys.VARIABLES, 'thresholds'],
            initializer=0.05)
    flat_x = tf.reshape(x, [-1])
    k = int(flat_x.get_shape().dims[0].value * (1 - p))
    topK, _ = tf.nn.top_k(tf.abs(flat_x), k)
    update_thre = thre.assign(topK[-1])
    tf.add_to_collection('update_thre_op', update_thre)

    mask = tf.zeros(shape)
    mask = tf.select((x > thre) | (x < -thre), tf.ones(shape), mask)

    with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
        w =  tf.sign(x) * tf.stop_gradient(mask)

    tf.histogram_summary(w.name, w)
    return w

def tw_ternarize(x, thre):

    shape = x.get_shape()

    thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thre)

    w_p = tf.get_variable('Wp', collections=[tf.GraphKeys.VARIABLES, 'positives'], initializer=1.0)
    w_n = tf.get_variable('Wn', collections=[tf.GraphKeys.VARIABLES, 'negatives'], initializer=1.0)

    tf.scalar_summary(w_p.name, w_p)
    tf.scalar_summary(w_n.name, w_n)

    mask = tf.ones(shape)
    mask_p = tf.select(x > thre_x, tf.ones(shape) * w_p, mask)
    mask_np = tf.select(x < -thre_x, tf.ones(shape) * w_n, mask_p)
    mask_z = tf.select((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)

    with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
        w =  tf.sign(x) * tf.stop_gradient(mask_z)

    w = w * mask_np

    tf.histogram_summary(w.name, w)
    return w


