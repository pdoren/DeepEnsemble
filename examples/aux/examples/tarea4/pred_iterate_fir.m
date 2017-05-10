function ypred = pred_iterate_fir(W1,B1,W2,B2,W3,B3,wi,xext)

ypred = NaN(size(xext));
for i = 1:length(xext)
    if i==1
        xtemp=wi;
    else
        xtemp=[xtemp(2:end) ypred(i-1)];
    end
     s=firnet3(W1,B1,W2,B2,W3,B3,xtemp);
     ypred(i) = s(end);
end
