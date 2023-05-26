source activate tls-env
python -m tls.run train \
       -c configs/hirid/GRU.gin \
       -l logs_new/main_exp/GRU_TLS/ \
       -t Dynamic_VentStatus_12Hours\
       -lr 3e-4\
       --hidden 128 \
       --do 0.0 \
       --depth 2 \
       --reg 'l1' \
       --reg-weight 10.0 \
       --smooth-type q_exp_param \
       --h-min 0 \
       --h-max 288 \
       --l-smooth 0.1 \
       -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000
