# python main.py --train_data ./Data/TrainSmall.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --feature_info_path ./Data/TrainSmall_FeatureInfo.NumBucket_100.yaml \
#                --log_dir ./Log\
#                --epochs 20 \
#                --optimizer Adam \
#                --lr 1e-4 \
#                --batch_size 1 \
#                --alpha 1.0 \
#                --exp_name lr_1e4_bs1_lambdaloss
# python main.py --train_data ./Data/TrainSmall80.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --feature_info_path ./Data/TrainSmall80_FeatureInfo.NumBucket_100.yaml \
#                --log_dir ./Log\
#                --epochs 20 \
#                --optimizer Adam \
#                --lr 1e-4 \
#                --batch_size 1 \
#                --alpha 1.0 \
#                --exp_name lr_1e4_bs1_lambdaloss
#
# python main.py --train_data ./Data/OFE.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --feature_info_path ./Data/OFE_FeatureInfo.NumBucket_100.yaml \
#                --log_dir ./Log\
#                --epochs 20 \
#                --optimizer Adam \
#                --lr 1e-3 \
#                --batch_size 1 \

# python main.py --train_data ./Data/OFE.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --feature_info_path ./Data/OFE_FeatureInfo.NumBucket_100.yaml \
#                --log_dir ./Log\
#                --epochs 20 \
#                --optimizer Adam \
#                --lr 1e-4 \
#                --batch_size 1 \

# python main.py --train_data ./Data/OFE.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --feature_info_path ./Data/OFE_FeatureInfo.NumBucket_100.yaml \
#                --log_dir ./Log\
#                --epochs 20 \
#                --optimizer Adam \
#                --lr 1e-5 \
#                --batch_size 1 \

# python main.py --train_data ./Data/OFE.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --feature_info_path ./Data/OFE_FeatureInfo.NumBucket_100.yaml \
#                --log_dir ./Log\
#                --epochs 20 \
#                --optimizer SGD \
#                --lr 1e-3 \
#                --batch_size 1 \

# python main.py --train_data ./Data/OFE.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --feature_info_path ./Data/OFE_FeatureInfo.NumBucket_100.yaml \
#                --log_dir ./Log\
#                --epochs 20 \
#                --optimizer SGD \
#                --lr 1e-4 \
#                --batch_size 1 \

# python main.py --train_data ./Data/OFE.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --feature_info_path ./Data/OFE_FeatureInfo.NumBucket_100.yaml \
#                --log_dir ./Log\
#                --epochs 20 \
#                --optimizer SGD \
#                --lr 1e-5 \
#                --batch_size 1 \
#
# python main.py --train_data ./Data/TrainSmall.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --feature_info_path ./Data/TrainSmall_FeatureInfo.NumBucket_100.yaml \
#                --log_dir ./Log\
#                --epochs 20 \
#                --optimizer SGD \
#                --lr 1e-5 \
#                --batch_size 1 

# python main.py --train_data ./Data/TrainSmall.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --feature_info_path ./Data/TrainSmall_FeatureInfo.NumBucket_100.yaml \
#                --log_dir ./Log\
#                --epochs 20 \
#                --optimizer SGD \
#                --lr 1e-4 \
#                --batch_size 1 

# python main.py --train_data ./Data/TrainSmall.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --feature_info_path ./Data/TrainSmall_FeatureInfo.NumBucket_100.yaml \
#                --log_dir ./Log\
#                --epochs 20 \
#                --optimizer SGD \
#                --lr 1e-3 \
#                --batch_size 1 

# python main.py --train_data ./Data/TrainSmall.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --feature_info_path ./Data/TrainSmall_FeatureInfo.NumBucket_100.yaml \
#                --log_dir ./Log\
#                --epochs 20 \
#                --optimizer Adam \
#                --lr 1e-3 \
#                --batch_size 1 

python main.py --train_data ./Data/TrainSmall.tsv \
               --test_data ./Data/EvalDataWithRating.tsv\
               --feature_info_path ./Data/TrainSmall_FeatureInfo.NumBucket_100.yaml \
               --log_dir ./Log\
               --epochs 20 \
               --optimizer Adam\
               --lr 1e-4 \
               --model_type NeuralNetEmbedding\
               --loss_type NDCG\
               --batch_size 1

