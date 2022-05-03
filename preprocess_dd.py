import os
from tqdm import tqdm

# PATHs
dd_path = "./data/ijcnlp_dailydialog/"
# output_path = os.path.join(dd_path, "pair_daily/")

# if not os.path.exists(output_path):
#     os.makedirs(output_path)

# Process
for split in ['train', 'validation', 'test']:
    dialogs = []
    data = []

    with open(os.path.join(dd_path, split, "dialogues_{}.txt".format(split))) as f:
        for line in f:
            dialogs.append(line.strip("__eou__").strip().split("__eou__"))

        for dial in tqdm(dialogs):
            for i in range(2, len(dial)):
                # Take everything upto i-2 (including)
                ctx = (" ".join(dial[:i-1])).strip()
                src = dial[i-1].strip()
                trg = dial[i].strip()
                if trg!="":
                    data.append((ctx, src, trg))

#     with open(os.path.join(dd_path, "{}.ctx.txt".format(split)), "w") as f:
#         for line in data:
#             f.write("{}\n".format(line[0]))

    with open(os.path.join(dd_path, "{}.src.txt".format(split)), "w") as f:
        for line in data:
            f.write("{}\n".format(line[0] + " " + line[1]))

    with open(os.path.join(dd_path, "{}.tgt.txt".format(split)), "w") as f:
        for line in data:
            f.write("{}\n".format(line[2]))

    print("Processed {}".format(split))

