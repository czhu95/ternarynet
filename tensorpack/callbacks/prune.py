import tensorflow as tf
import numpy as np

from ..utils import *
from .base import Callback

class PruneRunner(Callback):

    def _setup_graph(self):
        self._init_mask_op = tf.initialize_variables(tf.get_collection('masks'))
        self._init_thre_op = tf.initialize_variables(tf.get_collection('thresholds'))

    def _before_train(self):
        self._run_prune()
        sess = tf.get_default_session()
        sess.run(self._init_thre_op)
        sess.run(tf.get_collection('update_thre_op'))
        # import pdb; pdb.set_trace()

    def _trigger_epoch(self):
        sess = tf.get_default_session()
        sess.run(tf.get_collection('update_thre_op'))

    def _run_prune(self):
        sess = tf.get_default_session()
        # import pdb; pdb.set_trace()
        logger.info('Pruning weights.')
        # initialize mask variables to zeros.
        sess.run(self._init_mask_op)
        # run pruning operators.
        sess.run(tf.get_collection('update_mask_op'))
        sess.run(tf.get_collection('update_weight_op'))
