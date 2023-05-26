source activate tls-env
python -m tls.run train \
                             -c configs/hirid/transformer.gin \
                             -l logs/main_exp/multihorizon_transformer_cum/ \
                             -t Dynamic_RespFailure_2-22-2Hours\
                             --num-class 11 \
                             -lr 1e-4\
                             -bs 8\
                             --hidden 64 \
                             --do 0.0 \
                             --do_att 0.3 \
                             --depth 2 \
                             --heads 1 \
                             --reg 'l1' \
                             --reg-weight 10.0 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000

