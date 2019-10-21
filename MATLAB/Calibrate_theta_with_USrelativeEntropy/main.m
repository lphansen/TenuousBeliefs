%%%%%%%%%%%%%%%%%%%%%%%%
% Section 0 Set up
%%%%%%%%%%%%%%%%%%%%%%%%

clear;
param.q = 0.05;

param.alphayhat = 0.386;
param.alphazhat = 0;
param.betahat = 1;
param.kappahat = 0.019;
param.sigmay = [0.488;0];
param.sigmaz = [0.013;0.028];
param.delta = 0.002;

param.rho1 = 0;
param.rho2 = param.q^2 / norm(param.sigmaz)^2;
param.zbar = param.alphazhat / param.kappahat;
param.sigma = [param.sigmay.';param.sigmaz.'];
param.a = norm(param.sigmaz)^2 / det(param.sigma)^2 ;
param.b = - param.sigmay.'*param.sigmaz / det(param.sigma)^2 ;
param.d = norm(param.sigmay)^2 / det(param.sigma)^2 ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section 1 Solve for approximate dv
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bd = ApproxBound(param);
dvl = double(max(bd.nu));
dvr = double(min(bd.nu));
%dvl = ApproxBound(-1,param);
% dvr = ApproxBound(1,param);
%dvl = 1.20657417453607;
%dvr = 0.129461784426385;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section 2 Calibrate theta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%theta = linspace(0.075,0.15,20);
%theta = linspace(0.1,0.2,20); % for q = 0.05; rho2 not half
%theta = linspace(0.125,0.25,20); % for q = 0.1; rho2 half
% theta = linspace(0.175,0.325,20); % for q = 0.1; rho2 not half
% qUS = zeros(1,length(theta));
% parfor j = 1:length(theta)
%     qUS(j) = CalibrateTheta(theta(j),param,(dvl+dvr)/2,-2.5,2.5,dvl,dvr,0.01);
% end

tozero = @(theta)  CalibrateTheta(theta,param,(dvl+dvr)/2,-2.5,2.5,dvl,dvr,0.01,0.1); % 0.1 here is target qUS
out1.theta = fsolve(tozero,0.3);
tozero = @(theta)  CalibrateTheta(theta,param,(dvl+dvr)/2,-2.5,2.5,dvl,dvr,0.01,0.2); % 0.2 here is target qUS
out2.theta = fsolve(tozero,0.2);

% out2.theta = interp1(hl,theta,60);
% disp(['For HL 60, theta is ',num2str(out2.theta)]);
% out1.theta = interp1(hl,theta,120);
% disp(['For HL 120, theta is ',num2str(out1.theta)]);
outInf.theta = Inf;

[out2.hl,out2.v,out2.rh,out2.s1,out2.s2] = HL(out2.theta,param,0.65,-2.5,2.5,dvl,dvr,0.01,true);
[out1.hl,out1.v,out1.rh,out1.s1,out1.s2] = HL(out1.theta,param,0.55,-2.5,2.5,dvl,dvr,0.01,true);
[outInf.hl,outInf.v,outInf.rh,outInf.s1,outInf.s2] = HL(outInf.theta,param,0.45,-2.5,2.5,dvl,dvr,0.01,true);

out2 = Drift(out2,param);
out1 = Drift(out1,param);
outInf = Drift(outInf,param);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section 3 Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% beta-kappa Contour

[kappagrid,betagrid] =  meshgrid(0:0.001:0.06,0.5:0.005:1.3);
sigmainv = inv(param.sigma);
eta11 = sigmainv(1,1).*(betagrid-param.betahat) + sigmainv(1,2).*(param.kappahat-kappagrid);
eta12 = sigmainv(2,1).*(betagrid-param.betahat) + sigmainv(2,2).*(param.kappahat-kappagrid);
lhs = 0.5*(eta11.^2+eta12.^2) + (param.q.^2/norm(param.sigmaz)^2).*(-param.kappahat + param.sigmaz(1).*eta11 + param.sigmaz(2).*eta12);

figure('position',[100 100 800 800]);
contour(kappagrid,betagrid,lhs,[0 0]);

% % Choice of s1, s2 (In code r is s in paper; in code s is r in paper)

hold on;
zgrid = -2.5:0.01:2.5;
r = [outInf.s1;outInf.s2];
scatter(param.kappahat-r(2,:)./zgrid,param.betahat+r(1,:)./zgrid);
r = [out1.s1;out1.s2];
scatter(param.kappahat-r(2,:)./zgrid,param.betahat+r(1,:)./zgrid);
r = [out2.s1;out2.s2];
scatter(param.kappahat-r(2,:)./zgrid,param.betahat+r(1,:)./zgrid);

% % Choice of s1, s2

% figure('position',[100 100 800 800]);
% plot(out2.v.x,out2.s1,'color',[0 0.5 0],'LineWidth',2);
% hold on;
% plot(out1.v.x,out1.s1,'blue','LineWidth',2);
% plot(outInf.v.x,outInf.s1,'red','LineWidth',2);
% title('Choice of $s_1$','Interpreter','Latex');
% xlim([-0.5 0.5]);
% 
% figure('position',[100 100 800 800]);
% plot(out2.v.x,out2.s2,'color',[0 0.5 0],'LineWidth',2);
% hold on;
% plot(out1.v.x,out1.s2,'blue','LineWidth',2);
% plot(outInf.v.x,outInf.s2,'red','LineWidth',2);
% title('Choice of $s_2$','Interpreter','Latex');
% xlim([-0.5 0.5]);

% The value function
% 
% figure('position',[100 100 800 800]);
% plot(out2.v.x,out2.v.y(1,:),'color',[0 0.5 0],'LineWidth',2);
% hold on;
% plot(out1.v.x,out1.v.y(1,:),'blue','LineWidth',2);
% plot(outInf.v.x,outInf.v.y(1,:),'red','LineWidth',2);
% xlim([-1.5 1.5]);
% xlabel('z','FontSize',16);
% ylabel('$$ v(z) $$','Interpreter','Latex','FontSize',16);
% 
% figure('position',[100 100 800 800]);
% plot(out2.v.x,out2.v.y(2,:),'color',[0 0.5 0],'LineWidth',2);
% hold on;
% plot(out1.v.x,out1.v.y(2,:),'blue','LineWidth',2);
% plot(outInf.v.x,outInf.v.y(2,:),'red','LineWidth',2);
% xlim([-0.5 0.5]);
% xlabel('z','FontSize',16);
% ylabel('$$ \frac{dv}{dz} $$','Interpreter','Latex','FontSize',16);
% 
% figure('position',[100 100 800 800]);
% plot(out2.v.x(abs(out2.v.x)>0.005),out2.v.y(3,abs(out2.v.x)>0.005),'color',[0 0.5 0],'LineWidth',2);
% hold on;
% plot(out1.v.x,out1.v.y(3,:),'blue','LineWidth',2);
% plot(outInf.v.x,outInf.v.y(3,:),'red','LineWidth',2);
% xlim([-1.5 1.5]);
% xlabel('z','FontSize',16);
% ylabel('$$ \frac{d^2v}{dz^2} $$','Interpreter','Latex','FontSize',16);
% 
% % Ellipsoids of s
% % 
sd = sqrt(norm(param.sigmaz)^2/(2*param.kappahat));
z10 = norminv(0.1,param.alphazhat/param.kappahat,sqrt(norm(param.sigmaz)^2/(2*param.kappahat)));
z90 = norminv(0.9,param.alphazhat/param.kappahat,sqrt(norm(param.sigmaz)^2/(2*param.kappahat)));

figure('position',[100 100 800 800]);
DrawEllipsoids(z10,param,'red');
hold on;
%DrawEllipsoids(0,param,'black');
DrawEllipsoids(z90,param,'blue');
scatter(0,0,'black','*') %,'MarketEdgeColor','black','MarkerSize',10);
set(gca,'FontSize',16);
ax = gca;
ax.YTick = [-0.1 -0.05 0 0.05 0.1];
ax.XTick = [-0.01  -0.005 0 0.005 0.01];
xlim([-0.01 0.01]);
ylim([-0.1 0.1]);
ylabel('$r_1$','Interpreter','Latex','FontSize',20)
xlabel('$r_2$','Interpreter','Latex','FontSize',20) 

todraw.s1n = spline(out2.v.x,out2.s1,z10);
todraw.s2n = spline(out2.v.x,out2.s2,z10);
todraw.s1p = spline(out2.v.x,out2.s1,z90);
todraw.s2p = spline(out2.v.x,out2.s2,z90);
todraw.kn = spline(out2.v.x,out2.v.y(2,:),z10)*(-100);
todraw.kp = spline(out2.v.x,out2.v.y(2,:),z90)*(-100);

todraw.x = linspace(-0.05, 0.05);
plot(todraw.x,(todraw.x-todraw.s2n)*todraw.kn+todraw.s1n,'black','LineWidth',3);
plot(todraw.x,(todraw.x-todraw.s2p)*todraw.kp+todraw.s1p,'black','LineWidth',3);

% % Ellipsoids of r
% 
% % sd = sqrt(norm(param.sigmaz)^2/(2*param.kappahat));
% % figure('position',[100 100 800 800]);
% % DrawEllipsoidr(-sd,param,'red');
% % hold on;
% % DrawEllipsoidr(0,param,'black');
% % DrawEllipsoidr(sd,param,'blue');
% % scatter(0,0,'black','*') %,'MarketEdgeColor','black','MarkerSize',10);
% % set(gca,'FontSize',16);
% % ax = gca;
% % ax.YTick = [-0.2 -0.1 0 0.1 0.2];
% % ax.XTick = [-0.2 -0.1 0 0.1 0.2];
% % xlim([-0.2 0.2]);
% % ylim([-0.2 0.2]);
% % ylabel('$r_1$','Interpreter','Latex','FontSize',20)
% % xlabel('$r_2$','Interpreter','Latex','FontSize',20) 
% 

% % -H1 and -H2

% figure('position',[100 100 800 800]);
% plot(out2.v.x,-out2.rh(3,:),'color',[0 0.5 0],'LineWidth',2);
% hold on;
% plot(out1.v.x,-out1.rh(3,:),'blue','LineWidth',2);
% plot(outInf.v.x,-outInf.rh(3,:),'red','LineWidth',2);
% xlim([-0.5 0.5]);
% ylim([0 0.5]);
% plot([0 0],[0 0.5],'--k');
% xlabel('z','FontSize',16);
% ylabel('$$ -H_1 $$','Interpreter','Latex','FontSize',16);
% 
% figure('position',[100 100 800 800]);
% plot(out2.v.x,-out2.rh(4,:),'color',[0 0.5 0],'LineWidth',2);
% hold on;
% plot(out1.v.x,-out1.rh(4,:),'blue','LineWidth',2);
% plot(outInf.v.x,-outInf.rh(4,:),'red','LineWidth',2);
% xlim([-0.5 0.5]);
% ylim([0 0.5]);
% plot([0 0],[0 0.5],'--k');
% xlabel('z','FontSize',16);
% ylabel('$$ -H_2 $$','Interpreter','Latex','FontSize',16);

% % Drift Y
% 
% figure('position',[100 100 800 800]);
% plot(out2.v.x-param.zbar,out2.drifty,'color',[0 0.5 0],'LineWidth',3);
% hold on;
% plot(out1.v.x-param.zbar,out1.drifty,'blue','LineWidth',3);
% plot(outInf.v.x-param.zbar,outInf.drifty,'red','LineWidth',3);
% plot(outInf.v.x-param.zbar,param.alphayhat+param.betahat*(outInf.v.x-param.zbar),'black','LineWidth',3);
% xlim([-0.5 0.5]);
% ylim([-0.3 1]);
% set(gca,'FontSize',16);
% ax = gca;
% ax.XTick = [-0.50 -0.25 0.00 0.25 0.50];
% ax.YTick = [-0.30  0.00 0.30 0.60 0.90];
% plot([0 0],[-0.3 1],'--k','LineWidth',3);
% plot([-0.5 0.5],[0 0],'--k','LineWidth',3);
% xlabel('$$\bf z$$','Interpreter','Latex','FontSize',20);
% ylabel('$$ \bf \mu_Y $$','Interpreter','Latex','FontSize',20);

% Drift Z

figure('position',[100 100 800 800]);
plot(out2.v.x-param.zbar,out2.driftz,'color',[0 0.5 0],'LineWidth',3);
hold on;
plot(out1.v.x-param.zbar,out1.driftz,'blue','LineWidth',3);
plot(outInf.v.x-param.zbar,outInf.driftz,'red','LineWidth',3);
plot(outInf.v.x-param.zbar,param.alphazhat-param.kappahat*(outInf.v.x-param.zbar),'black','LineWidth',3);
xlim([-0.5 0.5]);
ylim([-0.025 0.01]);
set(gca,'FontSize',16);
ax = gca;
ax.XTick = [-0.50 -0.25 0.00 0.25 0.50];
ax.YTick = [-0.02 -0.01 0.00 0.01];
plot([0 0],[-0.025 0.01],'--k','LineWidth',3);
plot([-0.5 0.5],[0 0],'--k','LineWidth',3);
xlabel('$$\bf z$$','Interpreter','Latex','FontSize',20);
ylabel('$$\bf \mu_z $$','Interpreter','Latex','FontSize',20);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section 4 Entropies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
zgrid = -2.5:0.01:2.5;

tmp1 = ChernoffEntropyUS(out2.rh(3:4,:),out2.rh(1:2,:),zgrid,param);
tmp2 = ChernoffEntropyUS(out2.rh(3:4,:),repmat([0;0],1,size(zgrid,2)),zgrid,param);
tmp3 = ChernoffEntropyUS(out2.rh(1:2,:),repmat([0;0],1,size(zgrid,2)),zgrid,param);
tmp4 = ChernoffEntropyUS(out1.rh(3:4,:),out1.rh(1:2,:),zgrid,param);
tmp5 = ChernoffEntropyUS(out1.rh(3:4,:),repmat([0;0],1,size(zgrid,2)),zgrid,param);
tmp6 = ChernoffEntropyUS(out1.rh(1:2,:),repmat([0;0],1,size(zgrid,2)),zgrid,param);
tmp7 = ChernoffEntropyUS(outInf.rh(3:4,:),outInf.rh(1:2,:),zgrid,param);
tmp8 = ChernoffEntropyUS(outInf.rh(3:4,:),repmat([0;0],1,size(zgrid,2)),zgrid,param);
tmp9 = ChernoffEntropyUS(outInf.rh(1:2,:),repmat([0;0],1,size(zgrid,2)),zgrid,param);

disp(['theta for qUS = 0.2',' U to S Chernoff: ',num2str(tmp1)]);
disp(['theta for qUS = 0.2 ','U to B Chernoff: ',num2str(tmp2)]);
disp(['theta for qUS = 0.2 ','S to B Chernoff: ',num2str(tmp3)]);
disp(['theta for qUS = 0.2',' U to S HL: ',num2str(log(2)/tmp1)]);
disp(['theta for qUS = 0.2 ','U to B HL: ',num2str(log(2)/tmp2)]);

disp(['theta for qUS = 0.1 ','U to S Chernoff: ',num2str(tmp4)]);
disp(['theta for qUS = 0.1 ','U to B Chernoff: ',num2str(tmp5)]);
disp(['theta for qUS = 0.1 ','S to B Chernoff: ',num2str(tmp6)]);
disp(['theta for qUS = 0.2',' U to S HL: ',num2str(log(2)/tmp4)]);
disp(['theta for qUS = 0.2 ','U to B HL: ',num2str(log(2)/tmp5)]);

disp(['theta = Inf ','U to B Chernoff: ',num2str(tmp8)]);
disp(['theta = Inf ','S to B Chernoff: ',num2str(tmp9)]);
disp(['theta = Inf ','U to B HL: ',num2str(log(2)/tmp8)]);

disp(['theta for qUS = 0.2 ','U to S Relative: ',num2str(RelativeEntropyUS(out2.rh(3:4,:),out2.rh(1:2,:),zgrid,param))]);
disp(['theta for qUS = 0.2 ','U to B Relative: ',num2str(RelativeEntropyUS(out2.rh(3:4,:),repmat([0;0],1,size(zgrid,2)),zgrid,param))]);
disp(['theta for qUS = 0.2 ','S to B Relative: ',num2str(RelativeEntropyUS(out2.rh(1:2,:),repmat([0;0],1,size(zgrid,2)),zgrid,param))]);

disp(['theta for qUS = 0.1 ','U to S Relative: ',num2str(RelativeEntropyUS(out1.rh(3:4,:),out1.rh(1:2,:),zgrid,param))]);
disp(['theta for qUS = 0.1 ','U to B Relative: ',num2str(RelativeEntropyUS(out1.rh(3:4,:),repmat([0;0],1,size(zgrid,2)),zgrid,param))]);
disp(['theta for qUS = 0.1 ','S to B Relative: ',num2str(RelativeEntropyUS(out1.rh(1:2,:),repmat([0;0],1,size(zgrid,2)),zgrid,param))]);

disp(['theta = Inf ','U to S Relative: ',num2str(RelativeEntropyUS(outInf.rh(3:4,:),outInf.rh(1:2,:),zgrid,param))]);
disp(['theta = Inf ','U to B Relative: ',num2str(RelativeEntropyUS(outInf.rh(3:4,:),repmat([0;0],1,size(zgrid,2)),zgrid,param))]);
disp(['theta = Inf ','S to B Relative: ',num2str(RelativeEntropyUS(outInf.rh(1:2,:),repmat([0;0],1,size(zgrid,2)),zgrid,param))]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section 5 Doubld Check
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

U = out2.rh(3:4,:).';
S = out2.rh(1:2,:).';
disp(['theta for qUS = 0.2 ','E^U [|U-S|^2/2]: ',num2str(sqrt(StationaryExpect(U.',0.5*((U(:,1)-S(:,1)).^2+(U(:,2)-S(:,2)).^2),zgrid,param)*2)),'^2 / 2']);
disp(['theta for qUS = 0.2 ','E^U [|U|^2/2]: ',num2str(sqrt(StationaryExpect(U.',0.5*((U(:,1)).^2+(U(:,2)).^2),zgrid,param)*2)),'^2 / 2']);
[tmp1exp,tmp1den] = StationaryExpect(U.',0.5*((S(:,1)).^2+(S(:,2)).^2),zgrid,param);
disp(['theta for qUS = 0.2 ','E^U [|S|^2/2]: ',num2str(sqrt(tmp1exp*2)),'^2 / 2']);
[tmp2exp,tmp2den] = StationaryExpect(S.',0.5*((S(:,1)).^2+(S(:,2)).^2),zgrid,param);
disp(['theta for qUS = 0.2 ','E^S [|S|^2/2]: ',num2str(sqrt(tmp2exp*2)),'^2 / 2']);
[~,tmp3den] = StationaryExpect(zeros(size(S)).',0.5*((S(:,1)).^2+(S(:,2)).^2),zgrid,param);

tmp1 = StationaryExpect(U.',U(:,1),zgrid,param);
tmp2 = StationaryExpect(U.',U(:,2),zgrid,param);
tmp3 = RelativeEntropyUS(out2.rh(3:4,:),repmat([0;0],1,size(zgrid,2)),zgrid,param);
tmp4 = StationaryExpect(U.',(U(:,1)-tmp1).^2,zgrid,param);
tmp5 = StationaryExpect(U.',(U(:,2)-tmp2).^2,zgrid,param);
disp(['theta for qUS = 0.2 ','E^U [ U_1 ]: ',num2str(tmp1)]);
disp(['theta for qUS = 0.2 ','E^U [ U_2 ]: ',num2str(tmp2)]);
disp(['Implied by Relative Entropy E^U [ U_1^2 + U_2^2 ]: ', num2str(tmp3),'^2 = ', num2str(tmp3^2)]);
disp(['theta for qUS = 0.2 ','Var^U [ U_1 ]: ',num2str(tmp4)]);
disp(['theta for qUS = 0.2 ','Var^U [ U_2 ]: ',num2str(tmp5)]);
disp('Check that E^U [ U_1 ]^2 + E^U [ U_2 ]^2 + Var^U [ U_1 ] + Var^U [ U_2 ] = E^U [ U_1^2 + U_2^2 ]: ');
disp([num2str(tmp1), '^2 + ',num2str(tmp2), '^2 + ', num2str(tmp4), ' + ', num2str(tmp5), ' = ', ...
    num2str(tmp1^2+tmp2^2+tmp4+tmp5) , ' = ', num2str(tmp3^2)]);

% figure('position',[100 100 800 800]);
% plot(zgrid,tmp3den,'black','LineWidth',2);
% hold on;
% plot(zgrid,tmp1den,'blue','LineWidth',2);
% plot(zgrid,tmp2den,'red','LineWidth',2);
% plot(zgrid,(S(:,1).^2+S(:,2).^2)*0.5,'color',[0 0.5 0],'LineWidth',2);
% legend({'Z stationary density under baseline','Z stationary density under U', 'Z stationary density under S','$|S|^2/2$'},'Interpreter','Latex','FontSize',12);
% title('Z density and integrand for $q=0.1$ and $q_{u,s}=0.2$','Interpreter','latex','FontSize',20);

% U = out1.rh(3:4,:).';
% S = out1.rh(1:2,:).';
% disp(['theta for qUS = 0.1 ','E^U [|U-S|^2/2]: ',num2str(sqrt(StationaryExpect(U.',0.5*((U(:,1)-S(:,1)).^2+(U(:,2)-S(:,2)).^2),zgrid,param)*2)),'^2 / 2']);
% disp(['theta for qUS = 0.1 ','E^U [|U|^2/2]: ',num2str(sqrt(StationaryExpect(U.',0.5*((U(:,1)).^2+(U(:,2)).^2),zgrid,param)*2)),'^2 / 2']);
% [tmp1exp,tmp1den] = StationaryExpect(U.',0.5*((S(:,1)).^2+(S(:,2)).^2),zgrid,param);
% disp(['theta for qUS = 0.1 ','E^U [|S|^2/2]: ',num2str(sqrt(tmp1exp*2)),'^2 / 2']);
% [tmp2exp,tmp2den] = StationaryExpect(S.',0.5*((S(:,1)).^2+(S(:,2)).^2),zgrid,param);
% disp(['theta for qUS = 0.1 ','E^S [|S|^2/2]: ',num2str(sqrt(tmp2exp*2)),'^2 / 2']);
% [~,tmp3den] = StationaryExpect(zeros(size(S)).',0.5*((S(:,1)).^2+(S(:,2)).^2),zgrid,param);

% figure;
% plot(zgrid,tmp3den);
% hold on;
% plot(zgrid,tmp1den);
% plot(zgrid,tmp2den);
% plot(zgrid,(S(:,1).^2+S(:,2).^2)*0.5);

% U = outInf.rh(3:4,:).';
% S = outInf.rh(1:2,:).';
% disp(['theta = Inf ','E^U [|U-S|^2/2]: ',num2str(sqrt(StationaryExpect(U.',0.5*((U(:,1)-S(:,1)).^2+(U(:,2)-S(:,2)).^2),zgrid,param)*2)),'^2 / 2']);
% disp(['theta = Inf ','E^U [|U|^2/2]: ',num2str(sqrt(StationaryExpect(U.',0.5*((U(:,1)).^2+(U(:,2)).^2),zgrid,param)*2)),'^2 / 2']);
% [tmp1exp,tmp1den] = StationaryExpect(U.',0.5*((S(:,1)).^2+(S(:,2)).^2),zgrid,param);
% disp(['theta for qUS = 0.2 ','E^U [|S|^2/2]: ',num2str(sqrt(tmp1exp*2)),'^2 / 2']);
% [tmp2exp,tmp2den] = StationaryExpect(S.',0.5*((S(:,1)).^2+(S(:,2)).^2),zgrid,param);
% disp(['theta for qUS = 0.2 ','E^S [|S|^2/2]: ',num2str(sqrt(tmp2exp*2)),'^2 / 2']);
% [~,tmp3den] = StationaryExpect(zeros(size(S)).',0.5*((S(:,1)).^2+(S(:,2)).^2),zgrid,param);

% figure;
% plot(zgrid,tmp3den);
% hold on;
% plot(zgrid,tmp1den);
% plot(zgrid,tmp2den);
% plot(zgrid,(S(:,1).^2+S(:,2).^2)*0.5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section 6 Output RH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%dlmwrite('E:/ModelUncertainty/output/RH/rh_HStenuous39_q_dot05_fig3_qus_dot1.csv',out1.rh);
%dlmwrite('E:/ModelUncertainty/output/RH/rh_HStenuous39_q_dot05_fig3_qus_dot2.csv',out2.rh);
%dlmwrite('E:/ModelUncertainty/output/RH/rh_HStenuous39_q_dot05_fig3_qus_thetaInf.csv',outInf.rh);



