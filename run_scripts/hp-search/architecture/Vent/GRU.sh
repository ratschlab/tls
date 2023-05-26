source activate tls-env
python -m tls.run train \
       -c configs/hirid/GRU.gin \
       -l logs_new/hp_search_v2/architecture/GRU/ \
       -t Dynamic_VentStatus_12Hours\
       -lr 3e-4\
       --hidden 64 128 256\
       --do 0.0 \
       --depth 1 2 3 \
       --reg 'l1' \
       --reg-weight 0.0 0.01 0.1 1.0 10.0 100.0\
       -sd 1111 2222 3333 4444 5555\
