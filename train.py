#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Start model training.

   Author: Meng Cao
"""

from model.data_utils import REGDataset
from model.reg_model import REGShell
from model.config import Config

def main():
    # create instance of config
    config = Config(operation='train')

    # build model
    model  = REGShell(config)

    # create datasets
    train_set = REGDataset(config.filename_train, config=config, 
        max_iter=config.max_iter)
    dev_set = REGDataset(config.filename_dev, config=config, 
        max_iter=config.max_iter)
    sample_set = REGDataset(config.filename_sample, config=config, 
        max_iter=config.max_iter)

    # train model
    model.train(train_set, dev_set, sample_set)


if __name__ == "__main__":
    main()