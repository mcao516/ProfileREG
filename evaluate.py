#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate the model on the test set.

   Author: Meng Cao
"""

from model.data_utils import REGDataset
from model.reg_model import REGShell
from model.config import Config

def main():
    # create instance of config
    config = Config(operation='evaluate')

    # build model
    model = REGShell(config)
    model.restore_model('results/train/20180905_112821/model/checkpoint.pth.tar')

    # create datasets
    test = REGDataset(config.filename_test, config=config, 
        max_iter=config.max_iter)
    
    # evaluate on test set
    model.evaluate(test)


if __name__ == "__main__":
    main()



