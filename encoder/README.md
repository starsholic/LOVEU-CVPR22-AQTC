# Function-centric Encoder

1. Follow [LOVEU-CVPR22-AQTC/README.md](https://github.com/starsholic/LOVEU-CVPR22-AQTC/blob/main/README.md) to install required libraries and dataset.

2. Revise the dataset road of [LOVEU-CVPR22-AQTC/encoder/configs/vit_xlnet.yaml](https://github.com/starsholic/LOVEU-CVPR22-AQTC/blob/main/encoder/configs/vit_xlnet.yaml).

3. Run the script:

```
cd LOVEU-CVPR22-AQTC/encoder
```

For function-para generating and TF-IDF score calculating, first change the data_root in get_tfidf_score.py line 115, then,

```
python get_tfidf_score.py
```

For video encoding, 

```
python main.py --cfg configs/vit_xlnet.yaml FOR.VIDEO True DATASET.SPLIT "train" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_train" DATASET.LABEL "train_with_score.json"
python main.py --cfg configs/vit_xlnet.yaml FOR.VIDEO True DATASET.SPLIT "test" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_test" DATASET.LABEL "test_without_gt_with_score.json"
```

For script encoding, 

```
python main.py --cfg configs/vit_xlnet.yaml FOR.SCRIPT True DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_train"
python main.py --cfg configs/vit_xlnet.yaml FOR.SCRIPT True DATASET.SPLIT "test" DATASET.LABEL "test_without_gt_with_score.json" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_test"
```

For function-para encoding,

```
python main.py --cfg configs/vit_xlnet.yaml FOR.PARA True DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_train"
python main.py --cfg configs/vit_xlnet.yaml FOR.PARA True DATASET.SPLIT "test" DATASET.LABEL "test_without_gt_with_score.json" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_test"
```

For QA encoding,

```
python main.py --cfg configs/vit_xlnet.yaml FOR.QA True DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_train"
python main.py --cfg configs/vit_xlnet.yaml FOR.QA True DATASET.SPLIT "test" DATASET.LABEL "test_without_gt_with_score.json" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_test"
```