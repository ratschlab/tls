source activate tls-env
python -m tls.run train \
       -c configs/hirid/surv_GRU.gin \
       -l logs_new/main_exp/surv_GRU/ \
       -t Dynamic_CircFailure_12Hours\
       -o True \
       -lr 3e-4\
       -bs 32\
       --hidden 256 \
       --do 0.0 \
       --depth 2 \
       --reg 'l1' \
       --reg-weight 10.0 \
       --objective-type 'landmarking'\
       --h-max 1000 \
       -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000

