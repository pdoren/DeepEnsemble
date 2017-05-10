%
% cv_net        Trains a 2 layer network using backpropagation with cross-
% 		validation. The weights corresponding to the minimum 
%		crossvalidation error are returned along with the error history.
%
%		[MSE1,MSE1,cv,W1,B1,W2,B2,W3,B3] = 
%		     bp_ffnet2_cv(Inp1,Des1,Inp2,Des2,Param,W1,B1,W2,B2,W3,B3)
%		
%		Wi    - ith layer weight matrix
%		Bi    - ith layer bias vector
% 
%		Inp1  - Input vectors for training
%		Des1  - Desired vectors for training
%
%		Inp2  - Input vectors for validating
%		Des2  - Desired vectors for validating
%  
%		MSE1  - MSE for Training   (every epoch)
%		MSE2  - MSE for Validating (every Param[4] epochs)
%
%	        Param(1) = Maximum number of training epochs.
%   	        Param(2) = the step size or training rate, mu
%  	        Param(3) = 0 for iterative updates (stochastic)
%		         = 1 for block updating (true gradient)
% 	        Param(4) = number of epochs between validation evaluation.
% 	        Param(5) = Epochs past validation minimum to continue.
% 	        Param(6) = 1 on-line performance graphs
%
%		cv	 - epoch number where minimum validation occurs.
%
%		See also weight_init, bp_ffnet2, bp_firnet2, bp_firnet3.
%
%               Eric Wan. 
%               Oregon Graduate Institute. 4/19/96.


function [MSE1,MSE2,cv,W1cv,B1cv,W2cv,B2cv,W3cv,B3cv] = bp_ffnet2_cv(Inp1,Des1,Inp2,Des2,Param,W1,B1,W2,B2,W3,B3)


%======================================================================
% initialize variable

epochs        = Param(1);
validate_freq = Param(4);
GraphOn       = Param(6);
validate_stop = Param(5);

TP = [Param(4), Param(2), Param(3), -99];

[M1,N1] = size(Inp1);
N2      = length(Inp2);

% determine number of layers
if (nargin >= 10)
 if (W3 ~= []), Type = 3; end
end

if (Type==3),
 [Nodes,Taps] = architecture(M1,W1,W2,W3);
else
 [Nodes,Taps] = architecture(M1,W1,W2);
end


NetLen = length(Nodes);
SSTRT = sum(Taps)+1;
PASSES = epochs/validate_freq;  % epochs/validate_freq
MSE1 = zeros(epochs,Nodes(NetLen));
MSE2 = zeros(PASSES,Nodes(NetLen));
cv = validate_freq;
ct = 0;
mse_min = 9999;

%======================================================================

% Training

while (ct < PASSES) & (ct*validate_freq - cv < validate_stop),

 ct = ct+1;
 % train and test

 if (NetLen == 4),
   [W1,B1,W2,B2,W3,B3,mse_a]=bp_firnet3(W1,B1,W2,B2,W3,B3,Inp1,Des1,TP);
   y = firnet3(W1,B1,W2,B2,W3,B3,Inp2);

 elseif (Taps(2) == 1),

   [W1,B1,W2,B2,mse_a]=bp_ffnet2(W1,B1,W2,B2,Inp1,Des1,TP);
   y = ffnet2(W1,B1,W2,B2,Inp2);

 else

   [W1,B1,W2,B2,mse_a]=bp_firnet2(W1,B1,W2,B2,Inp1,Des1,TP);
   y = firnet2(W1,B1,W2,B2,Inp2);

 end


 % store error curves
 MSE1((ct-1)*validate_freq+1:ct*validate_freq,:) = mse_a';
 e = y - Des2;
 mse_b = sum(e(:,SSTRT:N2)'.^2)./(N2-SSTRT);
 MSE2(ct,:) = mse_b;


 % if at a minimum save current weights
 if (sum(mse_b) < mse_min),
   cv = ct*validate_freq;
   W1cv = W1;
   W2cv = W2;
   B1cv = B1;
   B2cv = B2;
   if (NetLen == 4),
    W3cv = W3;
    B3cv = B3;
   end
   mse_min = sum(mse_b);
 end
	
	
 % graph
 if (GraphOn==1) & (ct > 1),
  t = [(1:ct*validate_freq)];
  t2 =[(1:ct)*validate_freq];
  plot(t,MSE1(1:ct*validate_freq,:),'-',t2,MSE2(1:ct,:),'--')
  if (ct ==2), legend('-','train','--','validate'); end;
  pause(0.001);
 end

end

%
% truncate error curves
MSE1 = MSE1(1:ct*validate_freq,:);
MSE2 = MSE2(1:ct,:);





