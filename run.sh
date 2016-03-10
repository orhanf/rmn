#!/bin/bash

#export THEANO_FLAGS=device=gpu,floatX=float32
export THEANO_FLAGS=device=cpu,floatX=float32,mode=FAST_COMPILE 

python -m ipdb ${HOME}/git/rmn/train_rmn.py
