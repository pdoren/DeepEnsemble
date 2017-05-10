% setpath

close all
clear all
clc

% load laser sequence, create input and target symbols
y = load('laser.mat');
y= y.y;
S = (y' - mean(y'))/std(y');
SI = S(:,1:1000);
ST = S(:,1:1000);
SI2 = S(:,500:end);
ST2 = S(:,500:end);


net = rnn_new_elman(1,7,1);
mu = 0.0001;
neurons = 10;
% net = rnn_start_bp(net, 0.1, 0.0);
net = rnn_start_bptt(net, mu , 0.0, neurons);

% calculate NNL for untrained network (epoch 0)
[AO, AR] = rnn_sim(net, SI);
NNL = eval_mse(AO, ST);
fprintf('Epoch: %d, NNL: %f\n', 0, NNL(1));

% perform training and error calculation
AO1=SI;
for ep=(1:50),
    % [net, AO, AR] = rnn_train_bp(net, SI, ST, 0);
    [net, AO1, AR] = rnn_train_bptt(net, SI, ST, 0);
    [AO, AR] = rnn_sim(net, SI2);
    [e_mspe,NNL(ep+1)] = calc_error_time_series((AO - mean(AO))/std(AO),(ST2 - mean(ST2))/std(ST2));
    e_nmspe = NNL(ep+1);
    fprintf('Epoch: %d, NNL: %f\n', ep, NNL(ep+1));
    if NNL(ep+1)>NNL(ep)
        break
    end
end;


% plot chart
figure, plot([(AO - mean(AO))/std(AO);(ST2 - mean(ST2))/std(ST2)]');
legend({'Predicción';'Original'})
title('Validation set')
ypred_1step_val = AO;

TT = load('Laser Series from SFI Competition (Extended data).dat')';
TT = (TT-mean(TT))/std(TT);
[ypred_1step_ext, AR2] = rnn_sim(net, TT);
figure, plot([(ypred_1step_ext - mean(ypred_1step_ext))/std(ypred_1step_ext);TT]');
title('Test set')
legend({'Predicción';'Original'})
