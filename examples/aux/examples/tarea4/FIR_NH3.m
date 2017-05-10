% LASER - FIR


close all
clear all
clc
% ==================

%% cargar base de datos
TR = load('Laser Series from SFI Competition (Competition data).dat')';
TT = load('Laser Series from SFI Competition (Extended data).dat')';
TR = (TR - mean(TR))/std(TR);
TT = (TT - mean(TT))/std(TT);
figure,plot(1:1000,TR,'.-b',1001:1000+length(TT),TT,'g')
legend('Laser Competition Data','Laser Extended Data')

% Next, determine the length and dimension of the time series:
[Ni,Np] = size(TR);
inpT = TR(1:Np-1);   % The input series
desT = TR(2:Np);	 % The target series


% THE NETWORK ARCHITECTURE
% ========================

Nodes = [Ni 12 12 Ni];  % Number of nodes in each layer
Taps = [25 5 5];          % Number of time-delays
[W1,B1,W2,B2,W3,B3] = weight_init(Nodes,Taps);


% TRAINING PARAMETERS
% ===================

% Here we specify the number of times to train on the training set (epochs),
% and the learning rate or step size (mu).
epochs = 1000;
mu = 0.0001;

% We will use stochastic weight updates:
trainingMode = 0; 

% The MSE will be printed to screen on every 5th epoch:
disp_freq = 5;
% To disable architecture info before training, use disp_freq = -5
% To disable all printouts, use disp_freq = -(epochs+1)

% The training parameter vector combines the above parameters.
TP = [epochs mu trainingMode disp_freq];


% TRAINING
% ========

% The random initial weights, input and target series, and training
% parameters are passed to the training function; the returned values
% are the trained weights, and MSE over each training epoch:
inpT1 = inpT;
desT1 = desT;
[W1,B1,W2,B2,W3,B3,mse] = bp_firnet3(W1,B1,W2,B2,W3,B3,inpT1,desT1,TP);

% Here is a plot of the Mean Squared Error for each epoch.
figure,semilogy(mse,'b');


% PREDICTION
% ==========
%%
% The network can be used to predict the time series "one-step-ahead":
ypred_1step_val = pred_1step_fir(W1,B1,W2,B2,W3,B3,TR(end-sum(Taps)-501:end-501),TR(end-500:end));
ypred_1step_ext = pred_1step_fir(W1,B1,W2,B2,W3,B3,TR(end-sum(Taps):end),TT);

[e_mspe,e_nmspe] = calc_error_time_series(ypred_1step_val,TR(end-500:end))
figure,plot([ypred_1step_val;TR(end-500:end)]' )
legend('Predicted','Real')
title('Validation set')
figure,plot([ypred_1step_ext;TT]' )
legend('Predicted','Real')
title('Test set')
