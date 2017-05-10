%
% ffnet 	Feedforward neural network. Calls either ffnet2, firnet2, 
%               or firnet3.
%
%		y = ffnet(x,W1,B1,W2,B2,W3,B3)
%		
%		Wi    - ith layer weight matrix
%		Bi    - ith layer bias vector
% 
%		x     - Input vectors
%		y     - Output vectors
%
%		See also weight_init, ffnet2, firnet2, firnet3.
%
%               Eric Wan. 
%               Oregon Graduate Institute. 4/19/96.
%

function y = ffnet(x,W1,B1,W2,B2,W3,B3)


[n2,n1xt1] = size(W1);
[n3,n2xt2] = size(W2);
t2 = n2xt2/n2; %second layer taps


if (nargin > 5)
 if (W3 ~= []), Type = 3; end
end

 if (Type == 3) ,

   y = firnet3(W1,B1,W2,B2,W3,B3,x);

 elseif (t2 == 1),

   y = ffnet2(W1,B1,W2,B2,x);

 else

   y = firnet2(W1,B1,W2,B2,x);

 end





