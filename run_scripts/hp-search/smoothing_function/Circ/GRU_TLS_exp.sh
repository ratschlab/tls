source activate tls-env
python -m tls.run train \
       -c configs/hirid/GRU_TLS.gin \
       -l logs/ablations/TLS_hp_search/GRU_TLS/ \
       -t Dynamic_CircFailure_12Hours\
       -lr 3e-4\
       --hidden 256 \
       --do 0.0 \
       --depth 2 \
       --reg 'l1' \
       --reg-weight 10.0 \
       --smooth-type q_exp_param\
       --h-min 0 \
       --h-max 288\
       --l-smooth 0.01 0.05 0.1 0.2 0.5 1 2 5 \
       -sd 1111 2222 3333 4444 5555

