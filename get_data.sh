# DailyDialog
wget -q -c http://yanran.li/files/ijcnlp_dailydialog.zip
unzip ijcnlp_dailydialog.zip -d data/
rm ijcnlp_dailydialog.zip

unzip data/ijcnlp_dailydialog/test.zip -d data/ijcnlp_dailydialog/
unzip data/ijcnlp_dailydialog/train.zip -d data/ijcnlp_dailydialog/
unzip data/ijcnlp_dailydialog/validation.zip -d data/ijcnlp_dailydialog/

# DSTC-7
wget -c http://parl.ai/downloads/dstc7/dstc7_v2.tgz
mkdir -p data/dstc7-ubuntu
tar -xvf dstc7_v2.tgz \
        -C ./data/dstc7-ubuntu \
        ubuntu_dev_subtask_1.json \
        ubuntu_test_subtask_1.json \
        ubuntu_responses_subtask_1.tsv \
        ubuntu_train_subtask_1.json
rm dstc7_v2.tgz

