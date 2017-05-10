% demo1fir2.m %

clf;
figure(clf);
setfsize(500,200);
echo on
clc

% ======================================================================
%  DEMO1FIR2
% Time Series Prediction Using a Neural Network with Time-Delayed Inputs
%
% By: Alex T. Nelson, Eric A. Wan
%     Oregon Graduate Institute
%     11/22/95
% ======================================================================
%
% weight_init	 - Initializes the weights for a network with time delays.
% bp_firnet2 	 - Trains the two layer fir network with backpropagation.
% firnet2 	 - Simulates the two layer network with internal delays.

% Using the above functions, a network is used to predict
% a given time series.

pause % Strike any key to continue...
clc
% PREPARING THE DATA
% ==================

% We begin by loading the time series data from a file and putting it
% into row form (the kth column has data at time k):
load henonN.dat; 
henonN = henonN';

% Next, determine the length and dimension of the time series:
[Ni,Np] = size(henonN);

% For time series prediction, the kth target value should equal the
% (k+1)th input value. 
inpT = henonN(1:Np-1);   % The input series
desT = henonN(2:Np);	 % The target series

pause % Strike any key to continue...
clc
% THE NETWORK ARCHITECTURE
% ========================

% The dimension of the time series determines the number of input and 
% output nodes in the network. Here we arbitrarily choose 7 hidden nodes:
Nodes = [Ni 7 Ni];  % Number of nodes in each layer

% The following specifies that values of the input at times t, t-1, t-2,
% and t-3 (a total of 4 taps) will be used to drive the network at time t.
% The second layer will use a total of 3 taps.
Taps = [4 3];          % Number of time-delays

% We are now ready to create the random initial weight matrices for the 
% network:
[W1,B1,W2,B2] = weight_init(Nodes,Taps);

pause % Strike any key to continue...
clc
% TRAINING PARAMETERS
% ===================

% Here we specify the number of times to train on the training set (epochs),
% and the learning rate or step size (mu).
epochs = 75;
mu = 0.005;

% We will use stochastic weight updates:
trainingMode = 0; 

% The MSE will be printed to screen on every 5th epoch:
disp_freq = 5;
% To disable printouts during training, use disp_freq = -99

% The training parameter vector combines the above parameters.
TP = [epochs mu trainingMode disp_freq];

pause % Strike any key to start training...
clc
% TRAINING
% ========

% The random initial weights, input and target series, and training
% parameters are passed to the training function; the returned values
% are the trained weights, and MSE over each training epoch:

[W1,B1,W2,B2,mse] = bp_firnet2(W1,B1,W2,B2,inpT,desT,TP);

% Here is a plot of the Mean Squared Error for each epoch.
semilogy(mse,'b');

pause % Strike any key to continue...
clc
% PREDICTION
% ==========

% The network can be used to predict the time series "one-step-ahead":
[pred] = firnet2(W1,B1,W2,B2,inpT);

% For comparison the prediction is plotted in red, and the original 
% time series in yellow. Only 100 time points are shown.
P = [0 pred];  D = [inpT 0];
plot(P(1101:1200),'r')
hold on
plot(D(1101:1200),'y')
hold off

pause % Strike any key to plot the difference between the data and prediction.
clc
% Here's a plot of the difference between the original time series data
% and the one-step prediction.
plot(D-P,'g')

pause % Strike any key to continue...
echo off
disp('End of DEMO1FIR2')

 











