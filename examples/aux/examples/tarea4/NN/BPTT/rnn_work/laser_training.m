setpath

% load laser sequence, create input and target symbols
S = loadseq('laser.trs');
SI = S(:,1:8000-1);
ST = S(:,2:8000);
SI2 = S(:,8000-1:end-1);
ST2 = S(:,8000:end);

% create rnn = Elman's simple recurrent network with 6 input, 3 hidden and 6 output units
net = rnn_new_elman(4,6,4);

% net = rnn_start_bp(net, 0.1, 0.0);
% net = rnn_start_bptt(net, 0.1, 0.0, 1);
% net = rnn_start_rtrl(net, 0.5, 0.0);
net = rnn_start_ekf_bptt(net, 10, 1000, 0, 100, 0.00001, 1);
% net = rnn_start_ekf_rtrl(net, 1000, 0, 100, 0.00001, 1);

% calculate NNL for untrained network (epoch 0)
[AO, AR] = rnn_sim(net, SI);
NNL = eval_nnl(AO, ST);
fprintf('Epoch: %d, NNL: %f\n', 0, NNL(1));

% perform training and error calculation
for ep=(1:5),
    % [net, AO, AR] = rnn_train_bp(net, SI, ST, 0);
    % [net, AO, AR] = rnn_train_bptt(net, SI, ST, 0);
    % [net, AO, AR] = rnn_train_rtrl(net, SI, ST, 0);
    [net, AO, AR] = rnn_train_ekf_bptt(net, SI, ST, 0);
    % [net, AO, AR] = rnn_train_ekf_rtrl(net, SI, ST, 0);    

    [AO, AR] = rnn_sim(net, SI2);
    NNL(ep+1) = eval_nnl(AO, ST2);
    fprintf('Epoch: %d, NNL: %f\n', ep, NNL(ep+1));
end;

% plot chart
plot(NNL);