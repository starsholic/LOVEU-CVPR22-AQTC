# Q2A Encoder

1. Follow [LOVEU-CVPR22-AQTC/README.md](https://github.com/starsholic/LOVEU-CVPR22-AQTC/blob/main/README.md) to install required libraries and dataset.

2. Revise the dataset road of [LOVEU-CVPR22-AQTC/encoder/configs/vit_xlnet.yaml](https://github.com/starsholic/LOVEU-CVPR22-AQTC/blob/main/encoder/configs/vit_xlnet.yaml).

3. Run the script:

```
cd LOVEU-CVPR22-AQTC/encoder
```

For function-para generating and TF-IDF score calculating,

```
python get_tfidf_score.py
```

For video encoding, 

```
python main.py --cfg configs/vit_xlnet.yaml FOR.VIDEO True DATASET.SPLIT "train" DATASET.LABEL "train_with_para.json"
python main.py --cfg configs/vit_xlnet.yaml FOR.VIDEO True DATASET.SPLIT "test" DATASET.LABEL "test_without_gt_with_para.json"
```

For script encoding, 

```
python main.py --cfg configs/vit_xlnet.yaml FOR.SCRIPT True DATASET.SPLIT "train" DATASET.LABEL "train_with_para.json"
python main.py --cfg configs/vit_xlnet.yaml FOR.SCRIPT True DATASET.SPLIT "test" DATASET.LABEL "test_without_gt_with_para.json"
```

For function-para encoding,

```
python main.py --cfg configs/vit_xlnet.yaml FOR.PARA True DATASET.SPLIT "train" DATASET.LABEL "train_with_para.json"
python main.py --cfg configs/vit_xlnet.yaml FOR.PARA True DATASET.SPLIT "test" DATASET.LABEL "test_without_gt_with_para.json"
```

For QA encoding,

(1) Training set:

```
python main.py --cfg configs/vit_xlnet.yaml FOR.QA True DATASET.SPLIT "train" DATASET.LABEL "train_with_para.json"
```

(2) Testing set:

```
python main.py --cfg configs/vit_xlnet.yaml FOR.QA True DATASET.SPLIT "test" DATASET.LABEL "test_without_gt_with_para.json"
```