source activate tls-env
python -m tls.run train \
       -c configs/hirid/GRU.gin \
       -l logs/hp_search/architecture/GRU/ \
       -t  Dynamic_CircFailure_12Hours\
       -rs True\
       -lr 3e-4\
       --hidden 256 \
       --do 0.0 \
       --depth 2 \
       --reg 'l1' \
       --reg-weight 0.0 0.001 0.1 1.0 10.0 100.0\
       --depth 1 2 3 \
       -sd 1111 2222 3333 \
