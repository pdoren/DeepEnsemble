function neto = rnn_start_ekf_rtrl(net, Pp1, Pp2, Pr, Pq, lambda)
% RNN_START_EKF_RTRL - prepare rnn for EKF-RTRL training
% neto = rnn_start_ekf_rtrl(net, Pp1, Pp2, Pr, Pq, lambda)
% Pp1    - initial diagonal elements of P matrix
% Pp2    - initial off diagonal elements of P matrix
% Pr     - initial diagonal elements of R matrix
% Pq     - initial diagonal elements of Q matrix
% lambda - P matrix decay coef.

% set epoch persistent parameters 
P = ones(net.numWeights,net.numWeights) .* Pp2;
P(logical(eye(net.numWeights))) = Pp1;
net.ekf_rtrl.P = P;
net.ekf_rtrl.R = eye(net.numOutputUnits) .* Pr;
net.ekf_rtrl.Q = eye(net.numWeights) .* Pq;
net.ekf_rtrl.lambda = lambda;

% set weight changes
net.ekf_rtrl.DLT_W = zeros(1, net.numWeights);

% set RTRL derivatives (unit activities with respect of input, forward and
% recurrent weights in time T, T-1, ...)
net.ekf_rtrl.DA_DW = zeros(net.numAllUnits, net.numWeights, net.maxDelay+1);

% set initial activities
net.ekf_rtrl.actInit = net.actInit;


neto = net;
