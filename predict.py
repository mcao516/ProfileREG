#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Predict and write the results to file.

   Author: Meng Cao
"""

from model.data_utils import REGDataset
from model.reg_model import REGShell
from model.config import Config

import pickle


def write_prediction(filename, all_preds):
    """Write the prediction into file
       
       Args:
           filename: the file to write results
           preds: list of list, each list contains one or more(beam search)
           prediction expressions
    """
    with open(filename, 'w', encoding='utf-8') as file_object:
        for preds in all_preds:
            for pred in preds:
                file_object.write(pred+'\n')
            file_object.write('\n')

def main():
    # create instance of config
    config = Config(operation='predict')

    # build model
    model  = REGShell(config)
    model.restore_model('results/train/20181208_024721/model/checkpoint.pth.tar')

    # create datasets
    test = REGDataset(config.filename_test, config=config, 
        max_iter=config.max_iter)

    # evalution and write results into file
    all_preds, _, _, _, _, all_switch = model.predict(test)
    write_prediction(config.dir_output + 'preds.txt', all_preds)

    pickle.dump(all_switch, open(config.dir_output + 'switch.pickle', 'wb'))


if __name__ == "__main__":
    main()



