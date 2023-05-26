source activate tls-env
python -m tls.run train \
       -c configs/mimic3-benchmark/transformer.gin \
       -l logs/ablations/TLS_hp_search/transformer_TLS/ \
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
       --smooth-type q_sigmoid_param\
       --h-min 0 \
       --h-max 48\
       --l-smooth 0.05 0.1 0.2 0.5 1 2 5 \
       -sd 1111 2222 3333 4444 5555 \
