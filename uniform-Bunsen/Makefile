# ----------------------------------------------------------------------------
#
# Phiflow uniform Bunsen Makefile
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
# Reference

# training set
uniform-Bunsen-react-set: data_generation.py
	for i in `seq 0 2`; do \
		for j in `seq -1 1`; do \
			python $^ -o $@ -r 32 --bx 1 --er `echo "1 - ($$i)/10 " | bc -l` --amp `echo "15 - ($$i)*2.5 + ($$j)" | bc -l`  --gpu "-1" --seed 0; \
		done \
	done

# test set
uniform-Bunsen-react-testset: data_generation.py
	for i in `seq 0 1`; do \
		for j in `seq -1 1`; do \
			python $^ -o $@ -r 32 --bx 1 --er `echo "1 - (2*($$i) + 1) * 5/100 " | bc -l` --amp `echo "13 - ($$i)*2.5 + ($$j)" | bc -l`  --gpu "-1" --seed 0; \
		done \
	done

################################################################################
# Training models

# SOL-1
U-Bunsen-unet-sol1: hybrid_nn_pde_train_unet_MFP_v1.py # NOTE: don't test with "-n 1" (normalization problem because std(Re)=0!)
	python $^ --tf $@/tf --log $@/tf/run.log --epochs=100 --lr 0.0001 -l 100 -t 300 -m 1 -n 9 -b 3 --bx 1 --seed 0 --gpu '0' --cuda \
		--train uniform-Bunsen-react-set

################################################################################
# Run tests

# SOL-1
U-Bunsen-unet-sol1/run_test: hybrid_nn_pde_apply_v1.py
	for i in `seq 0 1`; do \
		 python $^ -o $@ \
		    --stats `dirname $@`/tf/dataStats.pickle \
            --model `dirname $@`/tf/model.h5 --gpu "-1" \
            --testset uniform-Bunsen-react-testset \
            -r 32 -b 3 --bIdx `echo "($$i) " | bc -l` -l 100 --er 1 -t 300; \
	done
