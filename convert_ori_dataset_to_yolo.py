# 将原始的 voc 格式的数据集转成yolov5训练要求的数据集格式
'''
dataset
    -images
        -train
            -xxx.jpg
        -test
        -val
    -labels
        -train
            -xxx.txt(label x,y,w,h)
        -test
        -val
'''
# coding:utf-8
from __future__ import print_function

import os
import random
import glob
import xml.etree.ElementTree as ET
from shutil import copyfile

def xml_reader(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return width, height, objects


def voc2yolo(filename, use_type):
    print(filename)
    width, height, objects = xml_reader(filename)

    lines = []
    for obj in objects:
        x, y, x2, y2 = obj['bbox']
        class_name = obj['name']
        label = classes_dict[class_name]
        cx = (x2 + x) * 0.5 / width
        cy = (y2 + y) * 0.5 / height
        w = (x2 - x) * 1. / width
        h = (y2 - y) * 1. / height
        line = "%s %.6f %.6f %.6f %.6f\n" % (label, cx, cy, w, h)
        lines.append(line)
        break

    txt_name = filename.replace(".xml", ".txt").replace("Annotations", "dataset/labels"+use_type)
    with open(txt_name, "w") as f:
        f.writelines(lines)

    # 将图片也拷贝一份到相应位置
    copyfile(filename.replace(".xml", ".jpg").replace("Annotations", "images"),
             filename.replace(".xml", ".jpg").replace("Annotations", "images").replace("images", "dataset/images" + use_type))

def imglist2file(xml_path_list):
    random.shuffle(xml_path_list)
    train_list = xml_path_list[:-100]
    valid_list = xml_path_list[-100:]
    with open("/home/zk/git_projects/hand_pose/hand_pose/ImageSets/Main/train.txt", "w") as f:
        f.writelines(train_list)
    with open("/home/zk/git_projects/hand_pose/hand_pose/ImageSets/Main/valid.txt", "w") as f:
        f.writelines(valid_list)

    for filename in train_list:
        voc2yolo(filename, '/train')
    for filename in valid_list:
        voc2yolo(filename, '/test')


if __name__ == "__main__":
    classes = ["four_fingers", "hand_with_fingers_splayed", "index_pointing_up", "little_finger", "ok_hand",
               "raised_fist", "raised_hand", "sign_of_the_horns", "three", "thumbup", "victory_hand"]
    classes_dict = {classes[i]: i for i in range(len(classes))}
    # print(classes_dict)
    xml_path = "../hand_pose/Annotations/*.xml"
    xml_path_list = glob.glob(xml_path)
    # print(xml_path_list)
    # img_path = "../hand_pose/images/*.jpg"
    # img_path_list = glob.glob(img_path)
    # print(len(img_path_list))
    # voc2yolo('../hand_pose/Annotations/index_pointing_up108.xml', '/train')
    imglist2file(xml_path_list)