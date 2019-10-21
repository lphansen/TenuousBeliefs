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

%zgrid = (-2.5):(0.002):2.5;
zgrid = (-2.5):(0.01):2.5;
T = 1000;
Dt = 0.1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section 1 Read in rh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rh = csvread('E:/ModelUncertainty/output/RH/rh_HStenuous39_q_dot05_fig3_qus_dot2.csv');
%rh = csvread('E:/ModelUncertainty/output/RH/rh_HStenuous2_q_dot1_fig3_120.csv');

drift = sigma * rh(3:4,:);
%muz = drift(2,:) + alphazhat - kappahat*zgrid(2:(end-1));
%muz = [0 muz 0];
muz = drift(2,:) + alphazhat - kappahat*zgrid;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section 2 Calculate Expectation of H1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%h1 = [0 rh(3,:) 0].';
h1 = rh(3,:).';
expectH1 = FeynmanKac(muz,sigmaz,zgrid,h1,T,Dt);

z10 = norminv(0.1,alphazhat/kappahat,sqrt(norm(sigmaz)^2/(2*kappahat)));
z90 = norminv(0.9,alphazhat/kappahat,sqrt(norm(sigmaz)^2/(2*kappahat)));

q10 = InterpQuantile(zgrid,expectH1,z10);
q90 = InterpQuantile(zgrid,expectH1,z90);
q50 = InterpQuantile(zgrid,expectH1,0);

figure('Position',[0,0,500,500]);
plot(0:Dt:T,-q10+0.01*sigmay(1),'r','LineWidth',2.5);
hold on;
plot(0:Dt:T,-q50+0.01*sigmay(1),'black','LineWidth',2.5);
plot(0:Dt:T,-q90+0.01*sigmay(1),'b','LineWidth',2.5);
%legend('z0 at .1 decile','z0 = 0','z0 at .9 decile','Location','southeast');
axis([0 40 0 0.32]);
%axis([0 1000 0 0.48]);
xlabel('Horizon(quarters)');
set(gca,'FontSize',18,'XTick',[0:5:40],'YTick',[0:0.08:0.32]);
%set(gca,'FontSize',18,'XTick',[0:200:1000],'YTick',[0:0.08:0.48]);
hold off;

% % Let's double check
% 
% tmpintl = zeros(length(zgrid),1);
% tmpintl(abs(zgrid)<1e-6,:) = 1/(zgrid(2)-zgrid(1));
% dencheck = KolmogorovF(muz,sigmaz,zgrid,tmpintl,T,Dt);
% 
% check0 = [];
% for t = 1:(T/Dt+1)
%     tmp = dencheck(:,t).*h1;
%     check0 = [check0 sum( (tmp(2:end)+tmp(1:(end-1)))/2*0.002 )];
% end
% 
% plot(0:Dt:T,check0)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section 3 Calculate Expectation of H2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%h2 = [0 rh(4,:) 0].';
h2 = rh(4,:).';
expectH2 = FeynmanKac(muz,sigmaz,zgrid,h2,T,Dt);

z10 = norminv(0.1,alphazhat/kappahat,sqrt(norm(sigmaz)^2/(2*kappahat)));
z90 = norminv(0.9,alphazhat/kappahat,sqrt(norm(sigmaz)^2/(2*kappahat)));

q10 = InterpQuantile(zgrid,expectH2,z10);
q90 = InterpQuantile(zgrid,expectH2,z90);
q50 = InterpQuantile(zgrid,expectH2,0);

figure('Position',[0,0,500,500]);
plot(0:Dt:T,-q10,'r','LineWidth',2.5);
hold on;
plot(0:Dt:T,-q50,'black','LineWidth',2.5);
plot(0:Dt:T,-q90,'b','LineWidth',2.5);
%legend('z0 at .1 decile','z0 = 0','z0 at .9 decile','Location','southeast');
axis([0 40 0 0.32]);
%axis([0 1000 0 0.48]);
xlabel('Horizon(quarters)');
set(gca,'FontSize',18,'XTick',[0:5:40],'YTick',[0:0.08:0.32]);
%set(gca,'FontSize',18,'XTick',[0:200:1000],'YTick',[0:0.08:0.48]);
hold off;

% % Let's double check

% tmpintl = zeros(length(zgrid),1);
% tmpintl(abs(zgrid)<1e-6,:) = 1/(zgrid(2)-zgrid(1));
% dencheck = KolmogorovF(muz,sigmaz,zgrid,tmpintl,T,Dt);
% 
% check0 = [];
% for t = 1:(T/Dt+1)
%     tmp = dencheck(:,t).*h2;
%     check0 = [check0 sum( (tmp(2:end)+tmp(1:(end-1)))/2*0.002 )];
% end
% 
% plot(0:Dt:T,check0)
