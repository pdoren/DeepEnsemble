function  spectra_gui(path)

data = importdata(path,' ',0);
if size(data,2) == 3
    part1 = 1;
else
    part1 = 0;
end
Ndata = 200;

t=data(1:Ndata,1);
x=data(1:Ndata,2);
if part1 ==1
    x_true = data(1:Ndata,3);
end


N=length(t);
x = x -mean(x);
dt=repmat(t,1,N);
dx=repmat(x,1,N);
dt= abs(dt-dt');
dx_c = dx.*dx';
dx_v=dx-dx';
sx=2*min(std(x))*power(N,-1/5);
Gx=exp(-0.5*dx_v.^2/sx^2);
Gx=Gx-mean(mean(Gx));

indx = triu(true(size(Gx)));
Gx=Gx(indx);
dx_c = dx_c(indx);
dt = dt(indx);

Nf = 2000;
f = linspace(0.3,1.3,Nf);
P=1/15;
MMm = 0;
if part1 == 0 %help the cursor and retain speed
    f = [f,linspace(0.35,0.351,20)];
    f = [f,linspace(0.7,0.702,20)];
    f = [f,linspace(0.482,0.483,20)];
    f = [f,linspace(1.24,1.241,20)];
    f = [f,linspace(0.812,0.8121,20)];
    f = sort(f,'ascend');
    if ( strcmp('MIRA',path(4:7)) ||  strcmp('LPV',path(4:6)) )
        f = linspace(0,0.02,Nf+100); %LPV/MIRA
    end
    if ( strcmp('MM',path(4:5))  )
        f = linspace(0.01,0.2,Nf+100); %Multimode
        MMm =1;
    end
    
end


% skewness rule
if ( strcmp('MM',path(4:5)) ||  strcmp('EB',path(4:5)) )
	st = 0.1;
else
	st = 0.4;
end
M = length(f);
N2 = length(dt);

if(part1==0 && MMm == 1)
    disp('Calculando CNMFS, por favor espere un momento');
    [lag,v,sigma,c] = slottedCorrentropy(t',x');
    [H,W,error,iter,cond] = NMF(lag',v',f,0.4,49,1,1,0,1,[]);
    H = H(1:length(f));
end

w = repmat(0.54 + 0.46*cos(pi*dt/max(dt)),1,M);
fr_basis = cos(2*pi*repmat(dt,1,M).*repmat(f,N2,1));
pk_basis = exp(-2*sin(pi*repmat(dt,1,M).*repmat(f,N2,1)).^2/st^2);

PSD =  transform(dx_c,w,fr_basis);
CSD = transform(Gx,w,fr_basis);
CKP = transform(Gx,w,pk_basis);

f1=figure;
figure(f1);
if part1 ==1
    subplot(3,1,1);plot(t,x,'kx',t,x_true,'b-');
    ylim([mean(x_true)-4*std(x_true) mean(x_true)+4*std(x_true)]);
    legend({'Contaminated','Underlying'});
else
    subplot(3,1,1);plot(t,x,'kx');
    ylim([mean(x)-4*std(x) mean(x)+4*std(x)]);
end

hs=subplot(3,1,2);
if part1 ==1
    plot(f,PSD/max(PSD),f,CSD/max(CSD)); 
elseif part1==0 && MMm ==1
    plot(f,PSD/max(PSD),f,CSD/max(CSD),f,CKP/max(CKP),f,H/max(H)); 
else
    plot(f,PSD/max(PSD),f,CSD/max(CSD),f,CKP/max(CKP)); 
end
dcm_obj = datacursormode;
set(dcm_obj,'UpdateFcn',@update_fold)
 xlabel('Frequency'); ylabel('Spectra'); 
if part1 ==1
    legend({'PSD','CSD'});
elseif part1==0 && MMm ==1
    legend({'PSD','CSD','CKP','CNMFS'});
else
    legend({'PSD','CSD','CKP'});
end
if ( strcmp('MIRA',path(4:7)) ||  strcmp('LPV',path(4:6)) )
   xlim([0 0.02]);
elseif ( strcmp('MM',path(4:5)) )
    xlim([0.01 0.2]);
else
     xlim([0.3 1.3]);
end

hf=subplot(3,1,3);Fold(hf,part1,data,P);

    text = uicontrol('Style','text',...
        'Position',[100 45 200 20],...
        'String',['Kernel Size: ',num2str(sx)]);
    
    uicontrol('Style', 'slider',...
        'Min',0.001,'Max',4.0,'Value',sx,...
        'Position', [100 20 200 20],...
        'Callback', {@change_ks,hs}); 
    
    function change_ks(hObj,event,ax) 
        sx = get(hObj,'Value');
        set(text,'String',['Kernel Size: ',num2str(sx)]);
        Gx=exp(-0.5*dx_v.^2/sx^2);
        Gx=Gx-mean(mean(Gx));
        indx = triu(true(size(Gx)));
        Gx=Gx(indx);
        CSD =  transform(Gx,w,fr_basis);
        
        if part1 ==1
            plot(ax,f,PSD/max(PSD),f,CSD/max(CSD));
            legend(ax,{'PSD','CSD'});
        elseif part1==0 && MMm ==0
            CKP =  transform(Gx,w,pk_basis);
            plot(f,PSD/max(PSD),f,CSD/max(CSD),f,CKP/max(CKP)); 
            legend(ax,{'PSD','CSD','CKP'});
        else
            CKP =  transform(Gx,w,pk_basis);
            plot(ax,f,PSD/max(PSD),f,CSD/max(CSD),f,CKP/max(CKP),H,H/max(H));
            legend(ax,{'PSD','CSD','CKP','CNMFS'});
        end
        xlabel(ax,'Frequency'); ylabel(ax,'Spectra');
        if ( strcmp('MIRA',path(4:7)) ||  strcmp('LPV',path(4:6)) )
           xlim(ax,[0 0.02]);
        elseif ( strcmp('MM',path(4:5)) )
            xlim(ax,[0.01 0.2]);
        else
             xlim(ax,[0.3 1.3]);
        end
        
    end
    function output = update_fold(hObj,event)
        P = 1/event.Position(1);
        Fold(hf,part1,data,P);
        output = {['Period: ',num2str(P,6)];['Frequency: ',num2str(1/P,6)]};

    end
end

function S = transform(x,w,basis)

    S = abs(sum(repmat(x,1,size(basis,2)).*w.*basis));
end
function Fold(axis,part1,data,P)

t = data(1:200,1);
x = data(1:200,2);
if part1 ==1
    x_true = data(1:200,3);
    xMean=mean(x_true);
    xSD=std(x_true);
else
    xMean=mean(x);
    xSD=std(x);
end

d=3.5;
phase=mod(t,P)/P;
[B,I]=sort(phase,'ascend');
mag=x(I);
plot(axis,[B,B+1],[mag,mag],'k.')
%errorbar(tOri,LC(:,2),LC(:,3),'kx')
set(axis,'YDir','reverse')
ylim(axis,[xMean-d*xSD xMean+d*xSD]);
xlabel(axis,['Phase, period: ',num2str(P)]);
ylabel(axis,'Magnitude');

end


