source activate tls-env
python -m tls.run train \
       -c configs/hirid/GRU.gin \
       -l logs/main_exp/multihorizon_GRU_cum/ \
       -t Dynamic_CircFailure_2-22-2Hours\
       -o True \
       -lr 3e-4\
       --num-class 11\
       --hidden 256 \
       --do 0.0 \
       --depth 2 \
       --reg 'l1' \
       --reg-weight 10.0 \
       -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000

