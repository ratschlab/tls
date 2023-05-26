source activate tls-env
python -m tls.run train \
       -c configs/mimic3-benchmark/transformer.gin \
       -l logs/main_exp/transformer_LS/ \
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
       --l-smooth 0.05\
       --smooth-type q_ls\
       -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000
