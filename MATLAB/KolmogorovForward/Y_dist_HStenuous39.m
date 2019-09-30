%%%%%%%%%%%%%%%%%%%%%%%%
% Section 0 Set up
%%%%%%%%%%%%%%%%%%%%%%%%

clear;
alphayhat = .386;
alphazhat = 0;
betahat = 1;
kappahat = .019;
sigmay = [.488;0];
sigmaz = [.013;.028];
delta = .002;
sigma = [sigmay.';sigmaz.'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section 1 Read in rh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rh = csvread('E:/ModelUncertainty/output/RH/rh_HStenuous39_q_dot1_fig3_qus_dot2.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section 2 Solve Kolmogorov for p(y,z) in the HL = 60
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define Grid

zgrid = (-2.5):(0.01):2.5;
ygrid = (-1):(0.002):1;

Nz = length(zgrid);
Ny = length(ygrid);
Dz = zgrid(2) - zgrid(1);
Dy = ygrid(2) - ygrid(1);
Dt = 0.001;

% Calculate drift

drift = sigma * rh(3:4,:);
muy = 0.01.*(drift(1,:) + alphayhat + betahat*zgrid);
%muy = 0.01.*(alphayhat + betahat*zgrid);
muy = repmat(muy.',1,Ny);
muz = drift(2,:) + alphazhat - kappahat*zgrid;
%muz = alphazhat - kappahat*zgrid;
muz = repmat(muz.',1,Ny);
sigmay = 0.01.*sigmay;

% Calculate initial distribution

pold = zeros(Nz,Ny);
pold(:,abs(ygrid-0)<1e-6) = normpdf(zgrid,alphazhat/kappahat,sqrt(norm(sigmaz)^2/(2*kappahat))) / Dy;

% Explicit FD

out = zeros(Nz,Ny,40/0.1);
for j = 1:(40/Dt)
    tmpy = muy .* pold;
    tmpz = muz .* pold;
    
    d_dy = ( tmpy(:,3:end) - tmpy(:,1:(end-2)) ) / (2*Dy);
    d_dz = ( tmpz(3:end,:) - tmpz(1:(end-2),:) ) / (2*Dz);
    d2_dy = ( pold(:,3:end) + pold(:,1:(end-2)) - 2*pold(:,2:(end-1)) ) / (Dy^2);
    d2_dz = ( pold(3:end,:) + pold(1:(end-2),:) - 2*pold(2:(end-1),:) ) / (Dz^2);
    d2_dydz = ( pold(3:end,3:end) - pold(1:(end-2),3:end) - pold(3:end,1:(end-2)) + pold(1:(end-2),1:(end-2)) ) / (4*Dy*Dz);
    
    rhs = - d_dy(2:(end-1),:) - d_dz(:,2:(end-1)) + norm(sigmay)^2/2.*d2_dy(2:(end-1),:) + norm(sigmaz)^2/2.*d2_dz(:,2:(end-1)) + (sigmay(1)*sigmaz(1)+sigmay(2)*sigmaz(2)).*d2_dydz;
    
    pnew = zeros(Nz,Ny);
    pnew(2:(end-1),2:(end-1)) = Dt*rhs + pold(2:(end-1),2:(end-1));
    
    tt = j*Dt;
    if mod(tt,0.1) == 0
        tt
        out(:,:,round(tt*10)) = pnew;
    end
    
    pold = pnew;
    
end    

% Plot: Time evolution

% for j = [20 50 100 200]
%     temp = out(:,:,j/10);
%     
%     subplot(2,1,1);
%     tstz = sum( (temp(:,1:(end-1))+temp(:,2:end))/2*Dy ,2);
%     plot(zgrid,tstz);
%     title('Kolmogorov forward solution for Z: HL 60');
%     legend('t=20','t=50','t=100','t=200');
%     hold on;
%     
%     subplot(2,1,2);
%     tsty = sum( (temp(1:(end-1),:)+temp(2:end,:))/2*Dz ,1);
%     plot(ygrid,tsty);
%     title('Kolmogorov forward solution for Y: HL 60');
%     legend('t=20','t=50','t=100','t=200');
%     hold on;
% end
% 
% hold off;

%save 'E:\ModelUncertainty\output\results\Kolmogorov_decile_qus_1_HStenuous39_sol.mat';

% Plot: Decile over t

ygrid0 = ygrid(2:end);
zgrid0 = zgrid(2:end);
q10y = 0;
q90y = 0;
q50y = 0;
q10z = norminv(0.1,alphazhat/kappahat,sqrt(norm(sigmaz)^2/(2*kappahat)));
q90z = norminv(0.9,alphazhat/kappahat,sqrt(norm(sigmaz)^2/(2*kappahat)));
q50z = norminv(0.5,alphazhat/kappahat,sqrt(norm(sigmaz)^2/(2*kappahat)));

for j = 1:(40/0.1) %(80/0.1)
    temp = out(:,:,j);
    tsty = sum( (temp(1:(end-1),:)+temp(2:end,:))/2*Dz ,1);
    disy = cumsum((tsty(2:end) + tsty(1:(end-1)))/2*Dy);
    tmp10 = interp1(disy(disy>0.001 & disy<0.999),ygrid0(disy>0.001 & disy<0.999),0.1);
    tmp90 = interp1(disy(disy>0.001 & disy<0.999),ygrid0(disy>0.001 & disy<0.999),0.9);
	tmp50 = interp1(disy(disy>0.001 & disy<0.999),ygrid0(disy>0.001 & disy<0.999),0.5);
    q10y = [q10y tmp10];
    q90y = [q90y tmp90];
    q50y = [q50y tmp50];
	
    tstz = sum( (temp(:,1:(end-1))+temp(:,2:end))/2*Dy ,2);
    disz = cumsum((tstz(2:end) + tstz(1:(end-1)))/2*Dz);
    tmp10 = interp1(disz(disz>0.001 & disz<0.999),zgrid0(disz>0.001 & disz<0.999),0.1);
    tmp90 = interp1(disz(disz>0.001 & disz<0.999),zgrid0(disz>0.001 & disz<0.999),0.9);
	tmp50 = interp1(disz(disz>0.001 & disz<0.999),zgrid0(disz>0.001 & disz<0.999),0.5);
    q10z = [q10z tmp10];
    q90z = [q90z tmp90];
    q50z = [q50z tmp50];	
end

%save('E:\ModelUncertainty\output\results\Kolmogorov_decile_qus_2_HStenuous39.mat','q10y','q50y','q90y','q10z','q50z','q90z');

% hold on;
% plot(0:0.01:10,q10,'b');
% hold on;
% plot(0:0.01:10,q90,'b');
% hold off;
% legend('Hat Model','HL=60','Location','northwest');
% title('Kolmogorov forward solution for Y, decile 0.1 and 0.9: Hat Model vs HL 60');
X = [0:0.1:40,fliplr(0:0.1:40)];
Y = [q10y(1:401)*100,fliplr(q90y(1:401)*100)];
fig = fill(X,Y,'red'); % or black (Use line 40 and 43 instead of 39 and 42)
set(fig,'facealpha',0.2);
set(fig,'EdgeColor','None');
hold on;
%plot(0:0.1:40,q50y(1:401)*100,'black','LineWidth',1.5);
%plot(0:0.1:40,q50y(1:401)*100,'red','LineStyle','--','LineWidth',1.5);
xlabel('Horizons(quarters)');
%title('Kolmogorov forward solution for Y, decile 0.1 and 0.9: HL 60');