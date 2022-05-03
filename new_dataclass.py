import json
import os
import random
import re

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

# import logging
# import coloredlogs
# from transformers import AutoTokenizer, BlenderbotTokenizer, \
#     BertTokenizer, BertTokenizerFast, \
#     RobertaTokenizer, RobertaTokenizerFast

from r1m_preprocess import filter_dialogs

C_MAX_LEN = 300
R_MAX_LEN = 60


# DailyDialog dataset loader class
class DialogData:
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, data_path, min_dial_len=3):
        super(DialogData, self).__init__()
        _file = data_path

        print("File:", _file)

        self.dial_data = []
        
        with open(_file) as f:
            for line in tqdm(f, desc="Loading data"):
                # if len(self.data) > max_items:
                #     break  # Enough for now
                Full_D = line.strip().strip("__eou__").split(" __eou__ ")
                self.dial_data.append(Full_D)
        
        self.min_dial_len = min_dial_len
        self.extract_cr_pairs()

    def extract_cr_pairs(self):
        MIN_DIAL_LEN = self.min_dial_len
        self.data = []
        ex_id = 0
        for Full_D in tqdm(self.dial_data, desc="Unrolling dialogs"):
            if len(Full_D) >= MIN_DIAL_LEN:
                # Ignoring the boundary to match with Ravi's test set.
                for j in range(MIN_DIAL_LEN, len(Full_D)+1):
                # for j in range(MIN_DIAL_LEN, len(Full_D) + 1):
                    D = Full_D[:j]
                    C = D[:-1]
                    R = D[-1].strip()
                    # mid = len(D)//2
                    # C = " __eou__ ".join(D[:mid])
                    # R = " __eou__ ".join(D[mid:])

                    self.data.append([ex_id, C, R, None])
                    ex_id += 1

        print("Loaded {} CR-samples.".format(len(self.data)))
        print("Samples:", self.data[random.randint(0, len(self.data))])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class SimpleDSTC7Data(DialogData):
        """An abstract class representing a Dataset.
        All other datasets should subclass it. All subclasses should override
        ``__len__``, that provides the size of the dataset, and ``__getitem__``,
        supporting integer indexing in range from 0 to len(self) exclusive.
        """

        def __init__(self, json_path, min_dial_len=2):
            """
            @param tokenizer: A huggingface tokenizer
            @param neg_per_positive: (npp) can be between 0 to 1 or any integer greater than 1.
            """
            # super(DialogData, self).__init__()
            _file = json_path

            self.dial_data = []
                    
            with open(_file) as f:
                raw_data = json.load(f)

                for i, full_log in enumerate(tqdm(raw_data, desc="Loading data")):
                    messages = full_log['messages-so-far']
                    dialog = []
                    last_sp = "who"
                    for turn in messages:
                        if turn["speaker"] != last_sp:
                            last_sp = turn["speaker"]
                            dialog.append(turn["utterance"])
                        else:
                            print(i)
                            dialog[-1] += (" " +turn['utterance'])
                    if 'options-for-correct-answers' in full_log:
                        dialog.append(full_log['options-for-correct-answers'][0]['utterance'])
                    self.dial_data.append(dialog)
                del raw_data

            # Filter URL and weird pieces of text (Do this before extracting CR pairs)
            _, self.dial_data = filter_dialogs(self.dial_data)
            
            # need this to match the testset with other libraries!
            self.min_dial_len = min_dial_len
            self.extract_cr_pairs()