close all, clear all, clc, format compact

addpath([pwd, '/sim']);                        %SIM Path
addpath([pwd, '/sim/multi']);     
 
[XAg, UAg] = InitUniverse();

U = NaN*ones(20,20,5000,5);

for ti=1:5000
    for i =1:20
        for j = 1:20

            U(i,j,ti,1) =  sqrt((XAg(ti,3*i-2)-XAg(ti,3*j-2)).^2 + (XAg(ti,3*i-1)-XAg(ti,3*j-1)).^2);
            U(i,j,ti,2) =  XAg(ti,3*i-2)-XAg(ti,3*j-2);
            U(i,j,ti,3) =  XAg(ti,3*i-1)-XAg(ti,3*j-1);
            U(i,j,ti,4) =  UAg(ti,2*i-1).*cos(XAg(ti,3*i)) - UAg(ti,2*i-1).*cos(XAg(ti,3*j));
            U(i,j,ti,5) =  UAg(ti,2*i-1).*sin(XAg(ti,3*i)) - UAg(ti,2*i-1).*sin(XAg(ti,3*j));
        end
    end

end

for i =1:20
    filename = sprintf('%s','/home/iansebas/World/Research/Panagou/python/data/sim/m',int2str(i),'.mat'); %Modify Path accordingly
    A = squeeze(U(i, :, : , :));
    save(filename,'A','-v7');
end