source activate tls-env
python -m tls.run train \
       -c configs/mimic3-benchmark/transformer.gin \
       -l logs/hp_search/focal_strength/transformer/ \
       -t decomp_24Hours\
       -bs 8\
       -lr 1e-4\
       --hidden 64 \
       --do 0.1\
       --do_att 0.3 \
       --depth 2 \
       --heads 1\
       --reg 'l1' \
       --reg-weight 0.1 \
       --lr-decay 0.99 \
       --loss-weight 0.5 0.25 0.1 0.05 0.025 \
       --gamma 5 2 1 0.5 0.2 0.1 0 \
       -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000
