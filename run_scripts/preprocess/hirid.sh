tls preprocess --dataset hirid \
               --hirid-data-root [path to source] #TODO User \
               --work-dir [path to output] #TODO User \
               --resource-path ./preprocessing/resources/ \
               --horizons 2 4 6 8 10 12 14 16 18 20 22 \
	             --nr-worker 8