% ARCHITECTURE  Find FIR neural network architecture.
%
%	[Nodes,Taps] = architecture(Nin,W1,W2,W3) 
%		
%	Nin   - number of input nodes in the network
%	Wi    - ith layer weight matrix
%	Nodes - Vector specifying network dimensions
%		Nodes(1) = # of inputs
%		Nodes(i) = # hidden units
%		Nodes(L) = # of outpus
%	Taps  - Vector Specifying order of tap delay line for each layer
%			
%		See also ???
%
%		Alex T. Nelson 4/16/96

function [Nodes,Taps] = architecture(Nin,W1,W2,W3) 

n1 = Nin;
[n2,n1xt1] = size(W1);
t1 = n1xt1/n1; 
Nodes = [n1 n2];
Taps = t1;

[n3,n2xt2] = size(W2);
t2 = n2xt2/n2;
Nodes = [Nodes n3];
Taps = [Taps t2];

if ((nargin > 3) & ~(W3 == [])),
  [n4,n3xt3] = size(W3);
  t3 = n3xt3/n3;
  Nodes = [Nodes n4];
  Taps = [Taps t3];
end
