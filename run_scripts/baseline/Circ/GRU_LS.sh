source activate tls-env
python -m tls.run train \
       -c configs/hirid/GRU.gin \
       -l logs/main_exp/GRU_LS/ \
       -t Dynamic_CircFailure_12Hours\
       -o True \
       -lr 3e-4\
       --hidden 256 \
       --do 0.0 \
       --depth 2 \
       --smooth-type q_ls\
       --reg 'l1' \
       --l-smooth 0.05\
       --reg-weight 10.0 \
       -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000

