import os
import json
import re
from tqdm import tqdm
from new_dataclass import *

# PATHs
dstc_path = "./data/dstc7-ubuntu/"
# output_path = os.path.join(dd_path, "pair_daily/")

# if not os.path.exists(output_path):
#     os.makedirs(output_path)

def extract_from_json(dev, true_responses=None):
    dev_stuff = []
    for item in tqdm(dev, desc="Extracting dialog data from .json"):
        dialog = item["messages-so-far"]
        con = []
        last_p = "none"

        # Context
        #########
        for turn in dialog:
            if turn["speaker"] == last_p:
                con[-1] = con[-1] + turn["utterance"]
            else:
                con.append(turn["utterance"])
                last_p = turn["speaker"]

        # Negative Samples + Positive/True candidate
        ############################################
        negative_samples = {}
        for option in item["options-for-next"]:
            negative_samples[option['candidate-id']] = option['utterance']

        # Get the true response
        ########################################
        if "options-for-correct-answers" in item:
            # Can't handle more than 1 true response
            # assert len(item["options-for-correct-answers"]) == 1
            gt = item["options-for-correct-answers"][0]["utterance"]
            gt_hash = item["options-for-correct-answers"][0]["candidate-id"]
        else:
            gt = true_responses[item["example-id"]]['text']
            gt_hash = true_responses[item["example-id"]]['hash']

        # Remove true candidate
        del negative_samples[gt_hash]
        # assert(len(negative_samples)) == 99
        dev_stuff.append((item["example-id"], con, gt, list(negative_samples.values())))
    # logger.debug(item)
    return dev_stuff


def write(dialogs, split):
    data = []

    for dial in tqdm(dialogs, desc="Creating CR pairs"):
        _, con, resp, _ = dial
        # ctx, src = con[-2:]
        # Use all prev. utt. for context
        ctx = " ".join(con[:-1])
        src = con[-1]
        trg = resp.strip()

        ctx = ctx.strip()
        src = src.strip()
        if trg!="":
            data.append((ctx, src, trg))

#     with open(os.path.join(dstc_path, "{}.ctx.txt".format(split)), "w") as f:
#         for line in data:
#             f.write("{}\n".format(line[0]))

    with open(os.path.join(dstc_path, "{}.src.txt".format(split)), "w") as f:
        for line in data:
            f.write("{}\n".format(line[0] + " " + line[1]))

    with open(os.path.join(dstc_path, "{}.tgt.txt".format(split)), "w") as f:
        for line in data:
            f.write("{}\n".format(line[2]))

    print("Processed {}".format(split))

# Process
for split in ['dev', 'train']:
    with open(os.path.join(dstc_path, f"ubuntu_{split}_subtask_1.json")) as f:
        raw_data = json.load(f)
    if split != 'test':
        dialogs = extract_from_json(raw_data)
    else:
        with open(os.path.join(dstc_path, "ubuntu_responses_subtask_1.tsv")) as f:
            test_responses = {}
            for line in f:
                id, hash, text = re.split("\t", line)
                test_responses[int(id)] = {'hash': hash, 'text': text}
        
        dialogs = extract_from_json(raw_data, test_responses)
    
    write(dialogs, split)
    
    del raw_data, dialogs #, data
    
# =====================
# Process the test data
# =====================
split = 'test'    
dial_path = "data/dstc7-ubuntu/ubuntu_test_subtask_1.json"
data = SimpleDSTC7Data(dial_path, min_dial_len=3)
write(data, split)