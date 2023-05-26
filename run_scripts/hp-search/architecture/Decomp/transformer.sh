source activate tls-env
python -m tls.run train \
       -c configs/mimic3-benchmark/transformer.gin \
       -l logs/hp_search/architecture/transformer/ \
       -t decomp_24Hours\
       -bs 8\
       -rs True\
       -lr 1e-4 3e-4 1e-5 3e-5\
       --hidden 32 64 128 256 \
       --do 0.0 0.1 0.2 0.3 0.4 \
       --do_att 0.0 0.1 0.2 0.3 0.4 \
       --depth 1 2 3 \
       --heads 1 2 4\
       --reg 'l1' \
       --reg-weight 0.0 0.001 0.1 1.0 10.0 \
       -sd 1111 2222 3333