#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(_get_exp_file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/coco_format_dataset"
        self.train_ann = "instances_train2021.json"
        self.val_ann = "instances_val2021.json"
        self.output_dir = "/home/featurize/YOLOX_outputs"

        self.num_classes = 3

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1
