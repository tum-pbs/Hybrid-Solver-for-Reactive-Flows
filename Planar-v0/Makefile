# ----------------------------------------------------------------------------
#
# Phiflow chemical reaction Makefile
# Copyright 2023 Nilam T, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Chemical Reaction examples
#
# ----------------------------------------------------------------------------

SHELL:=/bin/bash

################################################################################
# training set
planar-v0-react-set: data_generation.py
	for i in `seq 0 5`; do \
		python $^ -o $@ -r 32 -b 0.05 --er `echo "1 - ($$i)/10 " | bc -l` --gpu "-1" --seed 0; \
	done

# test set
planar-v0-react-testset: data_generation.py
	for i in `seq 0 3`; do \
		python $^ -o $@ -r 32 -b 0.05 --er `echo "1 - (2*($$i) + 1) * 5/100 " | bc -l` --gpu "-1" --seed 0; \
	done
################################################################################
# Training models

# SOL-2
P-v0-hyb-unet-sol2: hybrid_nn_pde_train_unet_v1.py
	python $^ --tf $@/tf --log $@/tf/run.log --epochs=100 --lr 0.0001 -l 100 -t 300 -m 2 -n 6 -b 3 --seed 0 --gpu '0' --cuda \
		--train planar-v0-react-set

P-v0-hyb-unet-sol16: hybrid_nn_pde_train_unet_v1.py
	python $^ --tf $@/tf --log $@/tf/run.log --epochs=200 --resume=100 --lr 0.0001 -l 100 -t 300 -m 32 -n 6 -b 3 --seed 0 --gpu '0' --cuda \
		--train planar-v0-react-set

################################################################################
# Run tests

# SOL-2
P-v0-hyb-unet-sol2/run_test: hybrid_nn_pde_apply_v1.py
	for i in `seq 0 1`; do \
		 python $^ -o $@ \
		    --stats `dirname $@`/tf/dataStats.pickle \
            --model `dirname $@`/tf/model.h5 --gpu "-1" \
            --testset planar-v0-react-testset \
            -r 32 -b 3 --bIdx `echo "($$i) " | bc -l` -l 100 -t 300; \
	done