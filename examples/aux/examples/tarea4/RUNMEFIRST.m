cd  NN/FIR/SOURCE/
mex bp_firnet3.c
mex firnet3.c
mex jacobW.c
mex jacobX.c
addpath(pwd);
cd ../../..


addpath('NN/BPTT/rnn_train');
addpath('NN/BPTT/rnn_work');
addpath('NN/BPTT/tools');
