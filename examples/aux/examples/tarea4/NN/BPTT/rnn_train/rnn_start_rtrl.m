function neto = rnn_start_rtrl(net, alpha, beta)
% RNN_START_RTRL - prepare rnn for RTRL training
% neto = rnn_start_rtrl(net, alpha, beta)
% neto   - output network
% alpha  - learning rate
% beta   - momentum rate

% set epoch persistent parameters (learning and momentum rate)
net.rtrl.alpha = alpha;
net.rtrl.beta = beta;

% set weight changes
net.rtrl.DLT_W = zeros(1, net.numWeights);

% set RTRL derivatives (unit activities with respect of input, forward and
% recurrent weights in time T, T-1, ...)
net.rtrl.DA_DW = zeros(net.numAllUnits, net.numWeights, net.maxDelay+1);

% set initial activities
net.rtrl.actInit = net.actInit;

neto = net;

