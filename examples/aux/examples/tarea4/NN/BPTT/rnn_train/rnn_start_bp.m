function neto = rnn_start_bp(net, alpha, beta)
% RNN_START_BP - prepare rnn for standard BP training
% neto = rnn_start_bp(net, alpha, beta)
% neto   - output network
% alpha  - learning rate
% beta   - momentum rate

% set epoch persistent parameters 
% (learning and momentum rate)
net.bp.alpha = alpha;
net.bp.beta = beta;

% set weight changes
net.bp.DLT_W = zeros(1, net.numWeights);

% set persistent activities
net.bp.saveDelay = net.maxDelay;
net.bp.saveAct = zeros(net.numAllUnits, net.bp.saveDelay);
net.bp.saveAct(net.numInputUnits+2:net.numAllUnits,end-net.maxDelay+1:end) = net.actInit;

neto = net;


