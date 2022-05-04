# Dialog System Baselines Using OpenNMT-py

## Requirements

Tested with

    - python 3.7.8
    - pytorch 1.10.2
    - OpenNMT-py 2.2.0

```bash
pip install OpenNMT-py
```

## Steps

### 1. Get Data

Run `bash get_data.sh`.

### 2. Preprocess to create parallel src-tgt files

```sh
python preprocess_dd.py
python preprocess_dstc7_ubuntu.py
```

#### 2.a Validate count

- DD: test.src.txt and test.tgt.txt should have 5740 lines.
- DSTC7: test.src.txt and test.tgt.txt should have 3247 lines.


### 3. Build Vocab

Run following:

```sh
onmt_build_vocab -config configs/dd_transformer_wordvocab_model.yaml -n_sample 1000000
onmt_build_vocab -config configs/dstc7_transformer_wordvocab_model.yaml -n_sample 500000
```

### 4. Training

* Open `configs/dd_transformer_wordvocab_model.yaml` and edit line numbers 64-66, to add more GPUs. 
* Adjust `batch_size: 8192` on line 52. (8192 usually works for 12 GB)
* Then run:
```sh
onmt_train -config configs/dd_transformer_wordvocab_model.yaml
onmt_train -config configs/dstc7_transformer_wordvocab_model.yaml
```

#### 5. Translate

```sh
onmt_translate -model data/ijcnlp_dailydialog/run/BEST_MODEL_PATH.pt -src data/ijcnlp_dailydialog/test.src.txt -output data/ijcnlp_dailydialog/run/transformer_wordvocab_best.txt -gpu 0 -verbose
onmt_translate -model data/dstc7-ubuntu/run/BEST_MODEL_PATH.pt -src data/dstc7-ubuntu/test.src.txt -output data/dstc7-ubuntu/run/transformer_wordvocab_best.txt -gpu 0 -verbose
```

#### 6. Reformat

Create the final json files using the following commands:

```sh
python reformat_to_json.py \
    -src data/ijcnlp_dailydialog/test.src.txt \
    -tgt data/ijcnlp_dailydialog/test.tgt.txt \
    -pred data/ijcnlp_dailydialog/run/transformer_wordvocab_best.txt

python reformat_to_json.py \
    -src data/dstc7-ubuntu/test.src.txt \
    -tgt data/dstc7-ubuntu/test.tgt.txt \
    -pred data/dstc7-ubuntu/run/transformer_wordvocab_best.txt
```