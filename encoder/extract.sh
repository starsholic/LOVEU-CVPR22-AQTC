# CUDA_VISIBLE_DEVICES=7 python main.py --cfg configs/vit_xlnet.yaml FOR.VIDEO True DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_train"
# CUDA_VISIBLE_DEVICES=7 python main.py --cfg configs/vit_xlnet.yaml FOR.VIDEO True DATASET.SPLIT "test" DATASET.LABEL "test_without_gt_with_score.json" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_test"

# CUDA_VISIBLE_DEVICES=7 python main.py --cfg configs/vit_xlnet.yaml FOR.SCRIPT True DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_train"
# CUDA_VISIBLE_DEVICES=7 python main.py --cfg configs/vit_xlnet.yaml FOR.SCRIPT True DATASET.SPLIT "test" DATASET.LABEL "test_without_gt_with_score.json" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_test"

# CUDA_VISIBLE_DEVICES=7 python main.py --cfg configs/vit_xlnet.yaml FOR.PARA True DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_train"
# CUDA_VISIBLE_DEVICES=7 python main.py --cfg configs/vit_xlnet.yaml FOR.PARA True DATASET.SPLIT "test" DATASET.LABEL "test_without_gt_with_score.json" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_test"

# CUDA_VISIBLE_DEVICES=7 python main.py --cfg configs/vit_xlnet.yaml FOR.QA True DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_train"
# CUDA_VISIBLE_DEVICES=7 python main.py --cfg configs/vit_xlnet.yaml FOR.QA True DATASET.SPLIT "test" DATASET.LABEL "test_without_gt_with_score.json" DATASET.ROOT "/data/wushiwei/data/assistq/assistq_test"