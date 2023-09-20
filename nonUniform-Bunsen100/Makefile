
SHELL:=/bin/bash

################################################################################
# Reference

# training set
nonUniform-Bunsen100-react-set: data_generation.py
	for i in `seq 0 11`; do \
		python $^ -o $@ -r 100 -t 500 --bx 1.5 --er 1 --seed `echo "($$i+5) " | bc -l` --gpu "-1"; \
	done

# test set
nonUniform-Bunsen100-react-testset: data_generation.py
	for i in `seq 12 23`; do \
		python $^ -o $@ -r 100 -t 500 --bx 1.0 --er 1 --seed `echo "($$i+5) " | bc -l` --gpu "-1"; \
	done

################################################################################
# SOL-2
NU-Bunsen100-unet-sol2: hybrid_nn_pde_train_MFP_unet.py
	python $^ --tf $@/tf --log $@/tf/run.log --epochs=100 --lr 0.0001 -l 100 -t 500 -m 2 -n 12 -b 4 --bx 1.5 --seed 0 --gpu '0' \
		--train nonUniform-Bunsen100-react-set

NU-Bunsen100-unet-sol32: hybrid_nn_pde_train_MFP_unet.py
	python $^ --tf $@/tf --log $@/tf/run.log --epochs=200 --resume 100 --lr 0.0001 -l 100 -t 500 -m 32 -n 12 -b 4 --bx 1.5 --seed 0 --gpu '0' \
		--train nonUniform-Bunsen100-react-set

################################################################################
NU-Bunsen100-unet-sol2/run_test: hybrid_nn_pde_apply.py
	for i in `seq 0 2`; do \
		 python $^ -o $@ \
		    --stats `dirname $@`/tf/dataStats.pickle \
            --model `dirname $@`/tf/model.h5 --gpu "-1" \
            --testset nonUniform-Bunsen100-react-testset \
            -r 100 -b 4 --bIdx `echo "($$i) " | bc -l` -l 100 --bx 1.5 --er 1 -t 500; \
	done

