source activate tls-env
python -m tls.run train \
       -c configs/hirid/transformer.gin \
       -l logs/hp_search/architecture/transformer/ \
       -t Dynamic_RespFailure_12Hours\
       -rs True\
       -lr 1e-4\
       -bs 8\
       --hidden 64 \
       --do 0.0 \
       --do_att 0.3 \
       --depth 2 \
       --heads 1 \
       --reg 'l1' \
       --reg-weight 0.0 0.001 0.1 1.0 10.0 100.0\
       -sd 1111 2222 3333 \

