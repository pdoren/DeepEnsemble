
function [H,W,error,iter,cond] = NMF(lag,V,f,ks_init,max_iter,debug,lambda,kronecker_dictionary,remove_coeff,W)
    %Parameters and allocation
    HperIter =10;
    WperIter =10;
    error = zeros(max_iter,1); 
    %Frequency dictionary
    s = ones(length(f),1)*ks_init;
    ds = zeros(size(s));
    %Create Dictionary 
    TF = sin(pi*repmat(lag,1,length(f)).*repmat(f,length(lag),1)).^2;
    if(isempty(W))
        sMatrix = repmat(s',size(TF,1),1).^2;
        W =exp(-2*TF./sMatrix);
        %Normalization
        maxw = ones(size(W));
        minw = exp(-2./sMatrix);
        W = (W-minw)./(maxw-minw);    
        %Impulsive noise dictionary
        if kronecker_dictionary ==1
            W = [W,eye(length(V),length(V))];
        end    
    end
        
    %Initiate H:
    H = W'*V;  
    %Create plots
    if(debug)        
        fhandle = figure;
        update_plot(fhandle,f,lag,V,W,H,s,1,[],[]);
    end
    iter =1;    
    %retained_coeff = [];
    updating_coeff = [];
    %Initial coefficient estimation
    H=sparse_coding(V,W,H,HperIter,lambda,[]);
    %NMF loop
    while(1) 
        %Update coefficients
        H=sparse_coding(V,W,H,HperIter,lambda,[]);
        %Ignore coeff for speed up, 
        if(remove_coeff ==1)
            updating_coeff = find(H> (max(H)-min(H))*0.01 + min(H));  
            %length(retained_coeff)
            %do not drop constant
            %if(~any(retained_coeff == 1))
            %    retained_coeff = [1,retained_coeff];
            %end
        end
        %Adapt dictionary
        [W,s,ds]=dictionary_learning(V,W,H,TF,s,ds,WperIter,find(updating_coeff<=length(f)));
        %Compute error
        error(iter) = sum((V-W*H).^2)/sum(V.^2);
        %Update plots
        if(debug)
            update_plot(fhandle,f,lag,V,W,H,s,iter,[],[]);
            %[hmax,fmax] = max(H(1:length(f)));
            %disp(['Iter: ',num2str(iter+1),' f:',num2str(f(fmax)),' Kernel size:',num2str(s(fmax)),' Error:',num2str(error(iter)),' Q:',num2str(hmax/median(H(1:length(f))))]);
            %disp(['updating %:',num2str(100*length(updating_coeff)/size(W,2))]);
        end        
        %Stopping condition
        etol = 0.005;                
        if(iter > 1)
            %abs(error(iter) - error(iter-1))/error(iter)
            if(iter >= max_iter )
                cond =1;
                break;
            %elseif( abs(error(iter) - error(iter-1))/error(iter) < etol)
            %    cond =2;
            %    break;
            end
        end
        iter = iter +1;
    end

    %movie2avi(mov, 'example.avi', 'compression', 'None');

end


function H=sparse_coding(V,W,H,max_iter,lambda,retained_coeff)
    Hold = H;
    if (isempty(retained_coeff))
        WtV = W'*V;
        WtW = W'*W;
    else
        WtV = W(:,retained_coeff)'*V;
        WtW = W(:,retained_coeff)'*W(:,retained_coeff);
    end
    %lambda = 1;
    if (~isempty(retained_coeff))
        H = H(retained_coeff);
    end
    for iter =1:max_iter
        %H(1) =1;
        H=H.*(WtV./(WtW*H+lambda+eps));
        %H=H.*(W'*((V)./(W*H+eps)));
        %H(1) = mean(V);
    end
    if (~isempty(retained_coeff))
        Hold(retained_coeff) = H;
        H = Hold;
    end
end

function [W,sNew,dsNew]=dictionary_learning(V,W,H,TF,sOld,dsOld,max_iter,retained_coeff)
    
    mr = 0.5;
    lr = 0.01;
    Wlast = W;
    sLast = sOld;
    dsLast = dsOld;
    if (~isempty(retained_coeff))
        TF = TF(:,retained_coeff);
        W = W(:,retained_coeff);
        H = H(retained_coeff);
        sOld = sOld(retained_coeff);
        dsOld = dsOld(retained_coeff);
    end
    %tic
    
    for iter = 1:max_iter
        e = W*H-V;
        grad =  4*H(1:size(TF,2)).*sOld.^(-3).*sum(W(1:size(TF,1),1:size(TF,2))'.*TF'.*repmat(e',size(TF,2),1),2);
        sNew = (sOld + mr*dsOld- lr*grad);
        %sNew = (sOld - lr*grad);
        sNew = max(sNew,0.01);
        dsNew = sNew - sOld;
        
        sOld = sNew;
        dsOld = dsNew;
        
        sMatrix = repmat(sNew',size(TF,1),1).^2;
        Wpk = exp(-2*TF./sMatrix);
        maxw = ones(size(sMatrix));
        minw = exp(-2./sMatrix); 
        %minw(1) =0;
        W(1:size(TF,1),1:size(TF,2)) = (Wpk-minw)./(maxw-minw); 
    end
   % toc
    if (~isempty(retained_coeff))
        Wlast(:,retained_coeff) = W;
        W = Wlast;
        sLast(retained_coeff) = sNew;
        sNew = sLast;
        dsLast(retained_coeff) = dsNew;
        dsNew = dsLast;
        %TF = TF(:,retained_coeff);
        %W = W(:,retained_coeff);
        %H = H(retained_coeff);
        %sOld = sOld(retained_coeff);
        %dsOld = dsOld(retained_coeff);
    end

end

function update_plot(handle,f,lag,V,W,H,s,iter,CSD,CKP)
        
        figure(handle);
        gcf;
        axes('Position',[0.1 0.1 0.8 0.8]);
        set(gca,'nextplot','replacechildren') 
        subplot(3,1,1); 
        plot(f,H(1:length(f))/max(H(1:length(f))),'b-'); %hold on;
        %str = {'CNMFS','CSD','CKP'};
        %str = 'CNMFS';
        %if(~isempty(CSD))
        %    plot(f,CSD,'k-');
        %    str = {str,'CSD'};
        %end
        %if(~isempty(CKP))
         %   plot(f,CKP,'r-');
        %     str = {str;'CKP'};
        %end
        %hold off;
        %legend_handle= legend(str,'Fontsize',10);
        %set(legend_handle, 'Box', 'off')
        %set(legend_handle, 'Color', 'none')
        ylabel('CNMFS');  xlabel('Frequency'); title(['Iteration ',num2str(iter)]);
        axis tight
        subplot(3,1,2); plot(lag,V,lag,W*H); ylabel('Correntropy'); xlabel('Lag'); legend({'Original','Reconstruction'});
        subplot(3,1,3); plot(f,s); xlabel('Frequency'); ylabel('Kernel size')
end


