
SHELL:=/bin/bash

# training set
nonUniform-Bunsen32-react-set: data_generation.py
	for i in `seq 0 11`; do \
		python $^ -o $@ -r 32 -t 500 --bx 1.0 --er 1 --seed `echo "($$i+5) " | bc -l` --gpu "-1"; \
	done


# test set
nonUniform-Bunsen32-react-testset: data_generation.py
	for i in `seq 12 23`; do \
		python $^ -o $@ -r 32 -t 500 --bx 1.0 --er 1 --seed `echo "($$i+5) " | bc -l` --gpu "-1"; \
	done

################################################################################
# SOL-2
NU-Bunsen32-unet-sol2: hybrid_nn_pde_train_unet_MFP_v1.py
	python $^ --tf $@/tf --log $@/tf/run.log --epochs=100 --resume -1 --lr 0.0001 -l 100 -t 500 -m 2 -n 12 -b 4 --seed 0 --gpu '0' --cuda \
		--train nonUniform-Bunsen32-react-set

NU-Bunsen32-unet-sol32: hybrid_nn_pde_train_unet_MFP_v1.py
	python $^ --tf $@/tf --log $@/tf/run.log --epochs=200 --lr 0.0001 -l 100 -t 500 -m 32 -n 12 -b 4 --seed 0 --gpu '0' --resume 100 --cuda \
		--train nonUniform-Bunsen32-react-set

################################################################################
NU-Bunsen32-unet-sol2/run_test: hybrid_nn_pde_apply.py
	for i in `seq 0 2`; do \
		 python $^ -o $@ \
		    --stats `dirname $@`/tf/dataStats.pickle \
            --model `dirname $@`/tf/model.h5 --gpu "-1" \
            --testset nonUniform-Bunsen32-react-testset \
            -r 32 -b 4 --bIdx `echo "($$i) " | bc -l` -l 100 --er 1 -t 500; \
	done

NU-Bunsen32-unet-sol32/run_test: hybrid_nn_pde_apply.py
	for i in `seq 0 2`; do \
		 python $^ -o $@ \
		    --stats `dirname $@`/tf/dataStats.pickle \
            --model `dirname $@`/tf/model.h5 --gpu "-1" \
            --testset nonUniform-Bunsen32-react-testset \
            -r 32 -b 4 --bIdx `echo "($$i) " | bc -l` -l 100 --er 1 -t 500; \
	done
