# python main.py --train_data ./Data/TrainSmall.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --log_dir ./Log\
#                --epochs 20 \
#                --lr 1e-3 \
#                --batch_size 1 \
#                --alpha 1.0 \
#                --exp_name lr_1e3_bs1

# python main.py --train_data ./Data/TrainSmall.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --log_dir ./Log\
#                --epochs 20 \
#                --lr 1e-4 \
#                --batch_size 1 \
#                --alpha 1.0 \
#                --exp_name lr_1e4_bs1

# python main.py --train_data ./Data/TrainSmall.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --log_dir ./Log\
#                --epochs 20 \
#                --lr 1e-5 \
#                --batch_size 1 \
#                --alpha 1.0 \
#                --exp_name lr_1e5_bs1
# python main.py --train_data ./Data/TrainSmall.tsv \
#                --test_data ./Data/EvalDataWithRating.tsv \
#                --log_dir ./Log\
#                --epochs 20 \
#                --lr 1e-3 \
#                --batch_size 1 \
#                --alpha 1.0 \
#                --exp_name lr_1e3_bs1_uniform_001
python main.py --train_data ./Data/TrainSmall.tsv \
               --test_data ./Data/EvalDataWithRating.tsv \
               --log_dir ./Log\
               --epochs 150 \
               --lr 1e-4 \
               --batch_size 1 \
               --alpha 1.0 \
               --exp_name lr_1e4_bs1_lambdaloss
