"""
Train mnist, see more explanation at http://mxnet.io/tutorials/python/mnist.html
"""
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import mxnet as mx
import numpy as np
import cv2

DEBUG = True

def resize_image(img, target_size):
    #print img.shape
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_max)

    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    # pad to product of stride
    padded_im = np.zeros((target_size, target_size))
    padded_im[:img.shape[0], :img.shape[1]] = 255 - img
    return padded_im


def read_data(trainlist):
    """
    download and read data into numpy
    """
    images = []
    labels = []
    with open(trainlist) as f:
        trainset = f.readlines()
        for idx, trainsample in enumerate(trainset):
            imagefile, label = trainsample.strip().split(' ')
            img = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
            ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = resize_image(img, 28)
            if DEBUG:
                cv2.imwrite("%s" % "debug/train_image_chip_" + str(idx) + ".jpg", img)
            #img = cv2.resize(img, (28, 28))
            #print img.shape, img
            images.append(img)
            labels.append(int(label))

    labels = np.array(labels)
    images = np.array(images)
    #print labels.shape, images.shape

    return (np.array(labels), np.array(images))


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

def get_mnist_iter(args, kv):
    """
    create data iterator with NDArrayIter
    """
    (train_lbl, train_img) = read_data(args.trainlist)
    (val_lbl, val_img) = read_data(args.vallist)
    train = mx.io.NDArrayIter(
        to4d(train_img), train_lbl, args.batch_size, shuffle=True)
    val = mx.io.NDArrayIter(
        to4d(val_img), val_lbl, args.batch_size)
    return (train, val)


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-classes', type=int, default=11,
                        help='the number of classes')
    parser.add_argument('--num-examples', type=int, default=9180,
                        help='the number of training examples')
    parser.add_argument('--trainlist', type=str, default='idcardnum/train.txt',
                        help='trainlist')
    parser.add_argument('--vallist', type=str, default='idcardnum/val.txt',
                        help='vallist')
    fit.add_fit_args(parser)
    parser.set_defaults(
        # network
        network        = 'lenet',
        # train
        gpus           = None,
        batch_size     = 64,
        disp_batches   = 100,
        num_epochs     = 10,
        lr             = .01,
        lr_step_epochs = '10',
        model_prefix = 'model/digits'
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, get_mnist_iter)
