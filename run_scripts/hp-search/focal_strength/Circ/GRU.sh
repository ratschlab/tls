source activate tls-env
python -m tls.run train \
       -c configs/hirid/GRU.gin \
       -l logs/hp_search/focal_strength/GRU/ \
       -t Dynamic_CircFailure_12Hours\
       -o True \
       -lr 3e-4\
       --hidden 256 \
       --do 0.0 \
       --depth 2 \
       --reg 'l1' \
       --reg-weight 10.0 \
       --loss-weight 0.5 0.25 0.1 0.05 0.025 \
       --gamma 5 2 1 0.5 0.2 0.1 0 \
       -sd 1111 2222 3333 4444 5555

