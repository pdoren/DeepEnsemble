function neto = rnn_start_ekf_bptt(net, wsize, Pp1, Pp2, Pr, Pq, lambda)
% RNN_START_EKF_BPTT - prepare rnn for EKF-BPTT training
% neto = rnn_start_ekf_bptt(net, wsize, Pp1, Pp2, Pr, Pq, lambda)
% wsize  - BPTT window size
% Pp1    - initial diagonal elements of P matrix
% Pp2    - initial off diagonal elements of P matrix
% Pr     - initial diagonal elements of R matrix
% Pq     - initial diagonal elements of Q matrix
% lambda - P matrix decay coef.

% set epoch persistent parameters 
net.ekf_bptt.wsize = wsize;
P = ones(net.numWeights,net.numWeights) .* Pp2;
P(logical(eye(net.numWeights))) = Pp1;
net.ekf_bptt.P = P;
net.ekf_bptt.R = eye(net.numOutputUnits) .* Pr;
net.ekf_bptt.Q = eye(net.numWeights) .* Pq;
net.ekf_bptt.lambda = lambda;

% set weight changes
net.ekf_bptt.DLT_W = zeros(1, net.numWeights);

% set persistent activities
net.ekf_bptt.saveDelay = net.maxDelay + wsize - 1;
net.ekf_bptt.saveAct = zeros(net.numAllUnits, net.ekf_bptt.saveDelay);
net.ekf_bptt.saveAct(net.numInputUnits+2:net.numAllUnits,end-net.maxDelay+1:end) = net.actInit;

neto = net;
