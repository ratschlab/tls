source activate tls-env
python -m tls.run train \
       -c configs/hirid/GRU_TLS.gin \
       -l logs/hp_search/ls_strength/GRU_LS/ \
       -t Dynamic_CircFailure_12Hours\
       -o True \
       -lr 3e-4\
       --hidden 256 \
       --do 0.0 \
       --depth 2 \
       --smooth-type q_ls\
       --reg 'l1' \
       --l-smooth 0.0 0.01 0.05 0.1 0.2 0.3\
       --reg-weight 10.0 \
       -sd 1111 2222 3333 4444 5555

