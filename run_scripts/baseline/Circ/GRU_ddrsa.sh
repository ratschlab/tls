source activate tls-env
python -m tls.run train \
       -c configs/hirid/surv_GRU.gin \
       -l logs_new/main_exp_v3/surv_GRU_ddrsa/ \
       -t Dynamic_CircFailure_12Hours\
       -o True \
       -lr 3e-4\
       -bs 32\
       --hidden 256 \
       --do 0.0 \
       --depth 2 \
       --reg 'l1' \
       --reg-weight 10.0 \
       --objective-type 'ddrsa'\
       --h-max 1000 \
       --ad 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9\
       -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000

