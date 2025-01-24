CUDA_VISIBLE_DEVICES=0 python -u src/seq.py \
--data_dir ./data/sports/ \
--cuda \
--batch_size 32 \
--checkpoint ./out/sports/ \
--model_path sports_20x3_module_multihead_4_seed_1.pt

CUDA_VISIBLE_DEVICES=0 python -u src/topn.py \
--data_dir ./data/sports/ \
--cuda \
--batch_size 32 \
--checkpoint ./out/sports/ \
--model_path sports_20x3_module_multihead_4_seed_1.pt


CUDA_VISIBLE_DEVICES=0 python -u src/exp.py \
--data_dir ./data/sports/ \
--cuda \
--batch_size 32 \
--checkpoint ./out/sports/ \
--model_path sports_20x3_module_multihead_4_seed_1.pt
