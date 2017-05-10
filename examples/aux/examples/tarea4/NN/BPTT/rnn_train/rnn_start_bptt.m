function neto = rnn_start_bptt(net, alpha, beta, wsize)
% RNN_START_BPTT - prepare rnn for BPTT training
% neto = rnn_start_bptt(net, alpha, beta, wsize)
% neto   - output network
% alpha  - learning rate
% beta   - momentum rate
% wsize  - BPTT window size

% set epoch persistent parameters 
% (learning and momentum rate, BPTT window size)
net.bptt.alpha = alpha;
net.bptt.beta = beta;
net.bptt.wsize = wsize;

% set weight changes
net.bptt.DLT_W = zeros(1, net.numWeights);

% set persistent activities
net.bptt.saveDelay = net.maxDelay + wsize - 1;
net.bptt.saveAct = zeros(net.numAllUnits, net.bptt.saveDelay);
net.bptt.saveAct(net.numInputUnits+2:net.numAllUnits,end-net.maxDelay+1:end) = net.actInit;

neto = net;


