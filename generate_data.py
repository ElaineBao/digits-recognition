import cv2
import numpy as np
import os
import math
import numpy.random as npr

def generate_random_value(val, minvalue = 1):
    randvalue = int(np.random.random() * val)
    if randvalue % 2 ==0:
        randvalue = randvalue - 1
    if randvalue < minvalue:
        randvalue = minvalue
    return randvalue

def img_rotation(img, im_center, degree):
    im_width = im_center[0] * 2
    im_height = im_center[1] * 2
    matRotation = cv2.getRotationMatrix2D(im_center, degree, 1.0)
    angle = degree / 180.0 * math.pi
    im_width_rotated = int(abs(im_width * math.cos(angle)) + abs(im_height * math.sin(angle)))
    im_height_rotated = int(abs(im_height * math.cos(angle)) + abs(im_width * math.sin(angle)))
    matRotation[0, 2] += (im_width_rotated - im_width) / 2
    matRotation[1, 2] += (im_height_rotated - im_height) / 2
    img_rotated = cv2.warpAffine(img, matRotation, (im_width_rotated, im_height_rotated), borderValue=255)
    im_center_rotated = (im_width_rotated / 2, im_height_rotated / 2)

    return img_rotated, im_center_rotated


if __name__ == '__main__':

    img_dir = 'idcardnum'
    img_save_dir = os.path.join(img_dir, 'new')
    train_file = os.path.join(img_dir, 'train.txt')
    f = open(train_file, 'w')

    img_idx = 0
    for parent, dirnames, filenames in os.walk(img_dir):
        for idx, filename in enumerate(filenames):
            if '.png' in filename:
                # original image used for training
                filepath = os.path.join(parent, filename)
                img1 = cv2.imread(filepath, 0)
                label = filename.strip().split('_')[0]
                f.write("{} {}\n".format(filepath, label))

                # image with rotation
                ret, img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                im_height, im_width= img1.shape
                im_center = (im_width / 2, im_height / 2)
                degrees = npr.randint(0, 30, size=4)
                for degree in degrees:
                    img_rotated, im_center_rotated = img_rotation(img1, im_center, degree - 15)
                    filepath = os.path.join(img_save_dir, "%s" % (label + '_' + str(img_idx) + '.png'))
                    cv2.imwrite(filepath, img_rotated)
                    img_idx += 1
                    f.write("{} {}\n".format(filepath, label))

                # images with gaussian blur
                kx = 20; ky = 20
                a=generate_random_value(kx);b=generate_random_value(ky)
                print a,b
                img2 = cv2.GaussianBlur(img1, (a,b), 0)
                filepath = os.path.join(img_save_dir, "%s" % (label + '_' + str(img_idx) + '.png'))
                f.write("{} {}\n".format(filepath, label))
                cv2.imwrite(filepath, img2)
                img_idx += 1

                # images with gaussian blur and rotation
                degrees = npr.randint(0, 30, size=4)
                for degree in degrees:
                    img_rotated, im_center_rotated = img_rotation(img2, im_center, degree - 15)
                    filepath = os.path.join(img_save_dir, "%s" % (label + '_' + str(img_idx) + '.png'))
                    cv2.imwrite(filepath, img_rotated)
                    img_idx += 1
                    f.write("{} {}\n".format(filepath, label))