
clc
echo on

% ======================================================================
%
% demo_cv
%
% Time Series Prediction Using a Neural Network with Time-Delayed Inputs.
% Demonstrates automatically training a network using a validation set
% for early stopping.
%
% By: Eric A. Wan, Alex T. Nelson
%     Oregon Graduate Institute
%     4/18/96
% ======================================================================
%
% weight_init	 - Initializes the weights for a network with time delays.
% cv_net 	 - Trains a network using both a training and
%                  cross-validation set
% ffnet 	 - Simulates network.


pause % Strike any key to continue...
clc

% ======================================================================
%
% We begin by loading the time series data. In this case the
% famous sunspot series.

load sunspotsR.dat  
[Np,Ni] = size(sunspotsR);

pause % Strike any key to continue...
clc

% ======================================================================
%
% Here is a plot of the data where we have specified different
% years that will be used for training, validation and testing.

Val = 165;
Test = 220;

% Train:      1700      - 1700+Val-1
% validate:   1700+Val  - 1700+Test-1
% test:       1700+Test - 1994


echo off

% ======================================================================
%
% plot sunspot data
clg
sunspots = abs(sunspotsR);
plot((1700:1700+Val-1),sunspots(1:Val),(1700+Val:1700+Test-1),sunspots(Val+1:Test),...
(1700+Test:1994),sunspots(Test+1:Np));
axis([1700 1995 0 200]);
legend('Training','Validation set','Testing')
xlabel('Year')
title('Sunspot Numbers');


echo on
pause % Strike any key to continue...
clc

% ======================================================================
%
% For training purposes we choose to normalize the data and "de-rectify"
% the series.  This represents the approximate 22 year solar cycle which 
% is composed of two 11 magnetic cycles of alternating polarity.


echo off
% plot normalized sunspot data

xs = sunspotsR';
sigSR = std(xs)*6;
ms = mean(xs);
x = (xs-ms)/sigSR;

plot((1700:1700+Val-1),x(1:Val),(1700+Val:1700+Test-1),x(Val+1:Test))
legend('Training','Validation set')
xlabel('Year')
title('Normalized and derectified Sunspot Numbers');


echo on
pause % Strike any key to continue...
clc

% ======================================================================
%
% Next, we initialize a network:

Nodes = [1 3 2 1];
Taps = [8 2 2];

% 3 layer network with 1 input by 4 hidden by 3 hidden by 1 output.
% 8 input lag window, 3 taps in second layer, 3 taps in third layer.

% initialize weights

[W1,B1,W2,B2,W3,B3] = weight_init(Nodes,Taps);

pause % Strike any key to continue...
clc

%======================================================================
%
% Next, prepare data for input to cv_net. This is a prediction problem
% so the Desired is just a shift of the Input series.

% Input and Desired for training
Inp1 = x(1:Val);      
Des1 = x(2:Val+1);

% Input and Desired for validating to determine early stopping.
Inp2 = x(Val+1:Test);   
Des2 = x(Val+2:Test+1);

pause % Strike any key to continue...

clc

%======================================================================
%
% Next, set various training and validation parameters

max_epochs    = 5000;	  % maximum training epochs to train
mu            = 0.01;     % learning rate.
StochBatch    = 0;        % stochastic = 0  batch = 1
validate_freq = 5;        % number of epochs between checking validation set
validate_stop = 500;      % max number of epochs to continue training after 
			  %  last min on validation set was found.
GraphOn       = 1;        % option for displaying graphics during training

% set all parameters
Param = [max_epochs mu StochBatch validate_freq validate_stop GraphOn];


pause % Strike any key to continue...
clc

%======================================================================
%
% And finally, we train using the cv_net function
%

% cv_net returns:
%
%	MSE1  - MSE for Training   (every epoch)
%	MSE2  - MSE for Validating (every validate_freq epochs)
%	cv    - epoch number where minimum validation occurs.
%	Wi    - ith layer weight matrix at epoch cv.
%	Bi    - ith layer bias vector at epoch cv.
%
% Note, cv_net trains a standard 2 layer feedforward network or 
% an FIR 2 or 3 layer network as specified by the weight dimensions.

% Sit back, this may take a while. (Get a faster computer!)

[MSE1,MSE2,cv,W1,B1,W2,B2,W3,B3] = cv_net(Inp1,Des1,Inp2,Des2,Param,W1,B1,W2,B2,W3,B3);


clc

%======================================================================
%
% Were done! Here's a plot of the final error curves with the minimum
% epoch marked.
%

echo off

% plot final error results

NetLen = length(Nodes);
L1 = length(MSE1);
L2 = length(MSE2);
plot((1:L1),MSE1,'-',(1:L2)*validate_freq,MSE2,'--')
H = max( max(MSE1(cv,:)), max(MSE2(cv/validate_freq,:)));
axis([0 L1 0 2*H])
xlabel('epoch');
ylabel('MSE');
hold on
plot(cv*ones(Nodes(NetLen)),MSE1(cv,:),'*');
plot(cv*ones(Nodes(NetLen)),MSE2(cv/validate_freq,:),'*');
hold off
legend('-','train',',--','validate','*','min test'); 
s = sprintf('Final Training/Validation Error Curves.  Minimum found at epoch %d',cv);
title(s)

echo on
pause % Strike any key to continue...
clc

clc

%======================================================================
%
% To plot the final prediction corresponding to the weights that
% were returned we can use ffnet:

yp = ffnet(x,W1,B1,W2,B2,W3,B3);

% Note, ffnet calls either ffnet2, firnet2, or firnet3 as needed.
%
% Here's a plot of the final results.

echo off
% shift by one since output is single step prediction of input
yp = [0 yp(1:Np-1)]*sigSR +ms;
yp = abs(yp);                    % for the derectified data
e = yp - sunspots';              % error

% some standard error measures
sigS = 1535;
SSTRT= sum(Taps)+1;
arv(1) =  sum(e(SSTRT:221).^2)/((221-SSTRT+1)*sigS);
arv(2) =  sum(e(222:256).^2)/(35*sigS);
arv(3) =  sum(e(257:280).^2)/(24*sigS);
arv(4) =  sum(e(281:295).^2)/(15*sigS);
arv(5) =  sum(e(222:295).^2)/(75*sigS);
arv;

% plot predictions

plot((1700:1700+Test-1),sunspots(1:Test),'g',(1700+Test:1994),sunspots(Test+1:Np),'b',...
(1700:1700+Test-1),yp(1:Test),'g+',(1700+Test:1994),yp(Test+1:Np),'b+');
axis([1700 1995 0 200])
legend('g-','train+validation','b-','test','+','prediction')
s = sprintf('Single step prediction. MSE/1535 for years 1700-1920 is %4.3f, years 1921-1994 is %4.3f', arv(1),arv(5));
title(s)

















