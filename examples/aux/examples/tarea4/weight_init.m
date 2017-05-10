%
% weight_init	initializes the weights in a feedforward network
%
%		[W1,B1,W2,B2,W3,B3] = weight_init(Nodes,Taps)
%		
%		Wi    - ith layer weight matrix
%		Bi    - ith layer bias vector
%		Nodes - Vector specifying network dimensions
%			Nodes(1) = # of inputs
%			Nodes(i) = # hidden units
%			Nodes(L) = # of outpus
%		Taps  - Vector Specifying order of tap delay line for each layer
%			
%		EXAMPLE: [W1,B1,W2,B2] = weight_init([3,5,2])
%                        Returns the weights for a standard 2 layer network
% 			 with 3 inputs, 5 hidden units, and 2 outputs.
%
%		EXAMPLE: [W1,B1,W2,B2,W3,B3] = weight_init([4,5,5,2],[5])
%  	   		 Returns the weights of a 3 layer network with 4 input
%			 time series and assuming a window on the input of 5 taps.
%			 (Total of 4x5=20 actual inputs to a feedforward network.)
%
%		EXAMPLE: [W1,B1,W2,B2] = weight_init([3,5,2],[5,5])
%			 Returns the wights of a 2 layer FIR network with 5th order
%			 taps in each of the layers.
%
%		See also ???
%

function [W1,B1,W2,B2,W3,B3] = weight_init(nodes,taps)


if nargin>1,
  taps = [taps(:); ones(3,1)];
else
  taps = [ones(3,1)];
end;


    W1 = (rand(nodes(2),nodes(1)*taps(1))-0.5) *sqrt(3/(nodes(1)*taps(1)));
    B1 = (rand(nodes(2),1)-0.5) *sqrt(3/nodes(1));

% two layers
    if length(nodes) > 2,
       W2 = (rand(nodes(3),nodes(2)*taps(2))-0.5) *sqrt(3/(nodes(2)*taps(2)));
       B2 = (rand(nodes(3),1)-0.5) *sqrt(3/(nodes(2)*taps(2)));
    end

% three layers
    if length(nodes) > 3,
       W3 = (rand(nodes(4),nodes(3)*taps(3))-0.5) *sqrt(3/(nodes(3)*taps(3)));
       B3 = (rand(nodes(4),1)-0.5) *sqrt(3/(nodes(3)*taps(3)));
    end
        



