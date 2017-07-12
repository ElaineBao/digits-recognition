"""
Train mnist, see more explanation at http://mxnet.io/tutorials/python/mnist.html
"""
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import mxnet as mx
from collections import namedtuple
import numpy as np
import cv2
from config import config

DEBUG = False

def get_lenet(num_classes=11):
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=num_classes)
    # loss
    lenet = mx.symbol.softmax(data=fc2, name='cls_prob')

    return lenet


def findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    if cv2.__version__[0] == '2':
        contours2, hierarchy2 = cv2.findContours(img.copy(), mode, method)
    elif cv2.__version__[0] == '3':
        _, contours2, hierarchy2 = cv2.findContours(img.copy(), mode, method)
    return contours2, hierarchy2


def resize_image(img, target_size):
    #print img.shape
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_max)

    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    # pad to product of stride
    padded_im = np.zeros((target_size, target_size))
    padded_im[:img.shape[0], :img.shape[1]] = img
    return padded_im


def read_data(img_bgr):
    """
    download and read data into numpy
    - input:
        - imgpath
    - output:
        - image_chips: each digit in one image_chip
        - num_image_chips: how many digits are detected in the image. Detection is based on contour blur and detection.
    """
    image_chips = []

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img_gray, config.RECOGNITION.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = 255 - img  # change bg to black


# find digits contours
    kx = 3; ky = 3
    img = cv2.GaussianBlur(img, (kx, ky), 0)
    contours, _ = findContours(img)
    rects = []; sort_m = [] # sort_m is used for sort rects by their position
    for i, contour in enumerate(contours):
        x0, y0, x1, y1 = np.min(contour[:, :, 0]), np.min(contour[:, :, 1]), np.max(contour[:, :, 0]), np.max(
            contour[:, :, 1])
        rects.append([x0, y0, x1, y1])
        sort_m.append(x0)

    sorted_index = sorted(range(len(sort_m)), key=lambda k: sort_m[k])
    rects_sorted = []
    for idx in sorted_index:
        rects_sorted.append(rects[idx])

    # segment image to image_chips
    for i, rect in enumerate(rects_sorted):
        x0, y0, x1, y1 = rect
        image_chip = img[y0:y1, x0:x1]
        image_chip = resize_image(image_chip, 28)
        image_chips.append(image_chip)
        if DEBUG:
            cv2.imwrite("%s"%"image_chip_" + str(i)+".jpg", image_chip)

    num_image_chips = len(image_chips)
    image_chips = np.array(image_chips)

    return to4d(image_chips), num_image_chips


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255


def probs_to_idnumber(_probs):
	id_num = ''
	probs = []
	for prob in _probs:
		x = max(prob)
		if x < config.RECOGNITION.DIGITS_RECOG_THRESH:
			continue
		idx = np.where(prob == x)
		probs.append(x)
		idx = idx[0][0]
		if idx == 10:
			id_num += 'X'
		else:
			id_num += str(idx)

	return id_num, probs


def load_checkpoint(model_path):
    save_dict = mx.nd.load(model_path)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def load_model(model_path, batchsize):
    arg_params, aux_params = load_checkpoint(model_path)
    # load network
    lenet = get_lenet()
    mod = mx.mod.Module(lenet, label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (batchsize, 1, 28, 28))])
    mod.set_params(arg_params, aux_params)

    return mod


def threshold_digits(digits, probs, digits_min_bit, digits_max_bit):
	if len(digits) < digits_min_bit:
		digits_keep = ""
		probs_keep = []
		return digits_keep, probs_keep
	elif len(digits) > digits_max_bit:
		sorted_index = sorted(range(len(probs)), key=lambda k: probs[k], reverse = True)
		idx_keep = sorted_index[:digits_max_bit]
		digits_keep = ""
		probs_keep = []
		for i in xrange(len(digits)):
			if i in idx_keep:
				digits_keep += digits[i]
				probs_keep.append(probs[i])
		return digits_keep, probs_keep
	else:
		return digits, probs


def digits_predict(img, model_path, digits_min_bit, digits_max_bit):
	img, num_digits = read_data(img)
	array = mx.nd.array(img)
	Batch = namedtuple('Batch', ['data'])

	mod = load_model(model_path, num_digits)
	mod.forward(Batch([array]))
	pred = mod.get_outputs()[0].asnumpy()
	digits, probs = probs_to_idnumber(pred)

	return threshold_digits(digits, probs, digits_min_bit, digits_max_bit)




if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="test mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_path', type=str, help='test_image or folder of test images')
    parser.add_argument('--model_path', default='model/digits-0002.params',type=str)

    args = parser.parse_args()

    if os.path.isdir(args.img_path):
        for parent, dirnames, filenames in os.walk(args.img_path):
            for idx, filename in enumerate(filenames):
                if '.jpg' in filename:
                    filepath = os.path.join(parent, filename)
                    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                    id_num, probs = digits_predict(img, args.model_path, 5, 18)
                    print filename, id_num
    else:
        img = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
        id_num, probs = digits_predict(img, args.model_path, 5, 18)
        print args.img_path, id_num

