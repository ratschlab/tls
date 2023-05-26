
tls preprocess --dataset mimic3 \
               --mimic3-data-root [path to source] #TODO User \
               --work-dir [path to output] #TODO User \
               --resource-path ./preprocessing/resources/ \
               --horizons 4 8 12 16 20 24 28 32 36 40 44 \
               --mimic3-static-columns Height
