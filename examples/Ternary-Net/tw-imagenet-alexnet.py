#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: alexnet-dorefa.py
# Author: Yuxin Wu, Yuheng Zou ({wyx,zyh}@megvii.com)

import cv2
import tensorflow as tf
import argparse
import numpy as np
import multiprocessing
import msgpack
import os, sys

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from ternary import tw_ternarize

BATCH_SIZE = 32

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 224, 224, 3], 'input'),
                InputVar(tf.int32, [None], 'label') ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        image = image / 255.0

        # monkey-patch tf.get_variable to apply fw
        old_get_variable = tf.get_variable
        def new_get_variable(name, shape=None, **kwargs):
            v = old_get_variable(name, shape, **kwargs)
            # don't binarize first and last layer
            if name != 'W' or 'conv0' in v.op.name or 'fct' in v.op.name:
                return v
            else:
                return tw_ternarize(v, args.t)
        tf.get_variable = new_get_variable

        with argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
                argscope([Conv2D, FullyConnected], use_bias=False, nl=tf.identity):
            logits = (LinearWrap(image)
                .Conv2D('conv0', 96, 12, stride=4, padding='VALID')
                .apply(tf.nn.relu)
                .Conv2D('conv1', 256, 5, padding='SAME', split=2)
                .BatchNorm('bn1')
                .MaxPooling('pool1', 3, 2, padding='SAME')
                .apply(tf.nn.relu)

                .Conv2D('conv2', 384, 3)
                .BatchNorm('bn2')
                .MaxPooling('pool2', 3, 2, padding='SAME')
                .apply(tf.nn.relu)

                .Conv2D('conv3', 384, 3, split=2)
                .BatchNorm('bn3')
                .apply(tf.nn.relu)

                .Conv2D('conv4', 256, 3, split=2)
                .BatchNorm('bn4')
                .MaxPooling('pool4', 3, 2, padding='VALID')
                .apply(tf.nn.relu)

                .FullyConnected('fc0', 4096)
                .BatchNorm('bnfc0')
                .apply(tf.nn.relu)

                .FullyConnected('fc1', 4096)
                .BatchNorm('bnfc1')
                .apply(tf.nn.relu)
                .FullyConnected('fct', 1000, use_bias=True)())
        tf.get_variable = old_get_variable

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label, 1)
        nr_wrong = tf.reduce_sum(wrong, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))
        wrong = prediction_incorrect(logits, label, 5)
        nr_wrong = tf.reduce_sum(wrong, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(5e-6))
        add_moving_summary(cost, wd_cost)

        add_param_summary([('.*/W', ['histogram', 'rms'])])
        self.cost = tf.add_n([cost, wd_cost], name='cost')

def get_data(dataset_name):
    isTrain = dataset_name == 'train'
    ds = dataset.ILSVRC12(args.data, dataset_name, dir_structure='train', shuffle=isTrain)

    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16,16:-16,:]

    if isTrain:
        class Resize(imgaug.ImageAugmentor):
            def __init__(self):
                self._init(locals())
            def _augment(self, img, _):
                h, w = img.shape[:2]
                size = 224
                scale = self.rng.randint(size, 308) * 1.0 / min(h, w)
                scaleX = scale * self.rng.uniform(0.85, 1.15)
                scaleY = scale * self.rng.uniform(0.85, 1.15)
                desSize = map(int, (max(size, min(w, scaleX * w)),\
                    max(size, min(h, scaleY * h))))
                dst = cv2.resize(img, tuple(desSize),
                     interpolation=cv2.INTER_CUBIC)
                return dst

        augmentors = [
            Resize(),
            # imgaug.Rotation(max_deg=10),
            # imgaug.RandomApplyAug(imgaug.GaussianBlur(3), 0.5),
            # imgaug.Brightness(30, True),
            # imgaug.Gamma(),
            # imgaug.Contrast((0.8,1.2), True),
            imgaug.RandomCrop((224, 224)),
            # imgaug.RandomApplyAug(imgaug.JpegNoise(), 0.8),
            # imgaug.RandomApplyAug(imgaug.GaussianDeform(
            #     [(0.2, 0.2), (0.2, 0.8), (0.8,0.8), (0.8,0.2)],
            #     (224, 224), 0.2, 3), 0.1),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean_224),
        ]
    else:
        def resize_func(im):
            h, w = im.shape[:2]
            scale = 256.0 / min(h, w)
            desSize = map(int, (max(224, min(w, scale * w)),\
                                max(224, min(h, scale * h))))
            im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
            return im
        augmentors = [
            imgaug.MapImage(resize_func),
            imgaug.CenterCrop((224, 224)),
            imgaug.MapImage(lambda x: x - pp_mean_224),
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(12, multiprocessing.cpu_count()))
    return ds

def get_config():
    mod = sys.modules['__main__']
    basename = os.path.basename(mod.__file__)
    logdir = os.path.join('train_log',
                         'ter-imagenet-alexnet',
                         't-{}'.format(args.t))

    logger.set_logger_dir(logdir)

    import shutil

    shutil.copy(mod.__file__, logger.LOG_DIR)

    # prepare dataset
    data_train = get_data('train')
    data_test = get_data('val')

    lr = tf.Variable(1e-4, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=data_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-5),
        callbacks=Callbacks([
            StatPrinter(), ModelSaver(),
            #HumanHyperParamSetter('learning_rate'),
            ScheduledHyperParamSetter(
                'learning_rate', [(56, 2e-5), (64, 4e-6)]),
            InferenceRunner(data_test,
                [ScalarStats('cost'),
                 ClassificationError('wrong-top1', 'val-error-top1'),
                 ClassificationError('wrong-top5', 'val-error-top5')])
        ]),
        model=Model(),
        step_per_epoch=10000,
        max_epoch=100,
    )

def run_image(model, sess_init, inputs):
    pred_config = PredictConfig(
        model=model,
        session_init=sess_init,
        session_config=get_default_sess_config(0.9),
        input_var_names=['input'],
        output_var_names=['output']
    )
    predict_func = get_predict_func(pred_config)
    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16,16:-16,:]
    words = meta.get_synset_words_1000()

    def resize_func(im):
        h, w = im.shape[:2]
        scale = 256.0 / min(h, w)
        desSize = map(int, (max(224, min(w, scale * w)),\
                            max(224, min(h, scale * h))))
        im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
        return im
    transformers = imgaug.AugmentorList([
        imgaug.MapImage(resize_func),
        imgaug.CenterCrop((224, 224)),
        imgaug.MapImage(lambda x: x - pp_mean_224),
    ])
    for f in inputs:
        assert os.path.isfile(f)
        img = cv2.imread(f).astype('float32')
        assert img is not None

        img = transformers.augment(img)[np.newaxis, :,:,:]
        outputs = predict_func([img])[0]
        prob = outputs[0]
        ret = prob.argsort()[-10:][::-1]

        names = [words[i] for i in ret]
        print(f + ":")
        print(list(zip(names, prob[ret])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    parser.add_argument('--load', help='load a checkpoint, or a npy (given as the pretrained model)')
    parser.add_argument('--data', help='ILSVRC dataset dir', default='/ssd/dataset/imagenet')
    parser.add_argument('--run', help='run on a list of images with the pretrained model', nargs='*')
    parser.add_argument('--t', type=float, default=0.05)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.run:
        assert args.load.endswith('.npy')
        run_image(Model(), ParamRestore(np.load(args.load, encoding='latin1').item()), args.run)
        sys.exit()

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()
