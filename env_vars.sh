#!/bin/bash 

export CAFFE_ROOT=$HOME/caffe/distribute # Adjust this to the caffe git path
export CAFFE_EXE=$CAFFE_ROOT/bin/caffe.bin
export GIT_REP_PATH=$(git 'rev-parse' '--show-toplevel' | tr -d '\n') # Assume this file sits in a git rep and returns the first top level git folder 

export PYTHONPATH=$HOME/caffe/distribute/python:$GIT_REP_PATH/python:$PYTHONPATH # Add the python module to python path
export PYTHONHOME=/Applications/anaconda

export PATH=$CAFFE_ROOT/bin:$PATH # Let the console find caffe.bin 
