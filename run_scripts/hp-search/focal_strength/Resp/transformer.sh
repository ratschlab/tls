source activate tls-env
python -m tls.run train \
       -c configs/hirid/transformer.gin \
       -l logs/hp_search/focal_strength/transformer/ \
       -t Dynamic_RespFailure_12Hours\
       -lr 1e-4\
       -bs 8\
       --hidden 64 \
       --do 0.0 \
       --do_att 0.3 \
       --depth 2 \
       --heads 1 \
       --loss-weight 0.5 0.25 0.1 0.05 0.025 \
       --gamma 5 2 1 0.5 0.2 0.1 0 \
       --reg 'l1' \
       --reg-weight 10.0 \
       -sd 6666 7777 8888 9999 0000

