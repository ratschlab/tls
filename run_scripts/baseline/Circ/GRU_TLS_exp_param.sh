source activate tls-env
python -m tls.run train \
       -c configs/hirid/GRU.gin \
       -l logs/main_exp/GRU_TLS/ \
       -t Dynamic_CircFailure_12Hours\
       -o True \
       -lr 3e-4\
       --hidden 256 \
       --do 0.0 \
       --depth 2 \
       --smooth-type q_exp_param\
       --reg 'l1' \
       --reg-weight 10.0 \
       --h-min 0 \
       --h-max 288 \
       --l-smooth 0.2\
       -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000

