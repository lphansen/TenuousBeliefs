function q = RelativeEntropyUS(etaU,etaS,zgrid,param)
% Implement tenous_36 Appendix C.2 
% zgrid should be 1*Nz
% Note that U is the H in Distortion.m; S is the R in Distortion.m

% Read param

alphayhat = param.alphayhat;
alphazhat = param.alphazhat;
betahat = param.betahat;
kappahat = param.kappahat;
sigmay = param.sigmay;
sigmaz = param.sigmaz;

rho1 = param.rho1;
rho2 = param.rho2;
zbar = param.zbar;
sigma = param.sigma;

Dz = zgrid(2)-zgrid(1);
Nz = size(zgrid,2);

% LHS
    
Q = [];
for j = 2:(Nz-1)
    one = (sigmaz.'*etaU(:,j)+alphazhat-kappahat*zgrid(j))/(2*Dz) - 0.5*norm(sigmaz)^2/(Dz^2);
    two = norm(sigmaz)^2/Dz^2;
    three = -(sigmaz.'*etaU(:,j)+alphazhat-kappahat*zgrid(j))/(2*Dz) - 0.5*norm(sigmaz)^2/(Dz^2);
    row = [zeros(1,j-2),one,two,three,zeros(1,Nz-j-1)];
    Q = [Q;row];
end
   
one = (sigmaz.'*etaU(:,1)+alphazhat-kappahat*zgrid(1))/Dz - 0.5*norm(sigmaz)^2/(Dz^2);
two = -(sigmaz.'*etaU(:,1)+alphazhat-kappahat*zgrid(1))/Dz + norm(sigmaz)^2/(Dz^2);
three = -0.5*norm(sigmaz)^2/(Dz^2);
row1 = [one,two,three,zeros(1,Nz-3)];
    
three = -(sigmaz.'*etaU(:,Nz)+alphazhat-kappahat*zgrid(Nz))/Dz - 0.5*norm(sigmaz)^2/(Dz^2);
two = (sigmaz.'*etaU(:,Nz)+alphazhat-kappahat*zgrid(Nz))/Dz + norm(sigmaz)^2/(Dz^2);
one = -0.5*norm(sigmaz)^2/(Dz^2);
rowNz = [zeros(1,Nz-3),one,two,three];
    
Q = [row1;Q;rowNz];

% RHS

tmp = etaU-etaS;
%tmp = etaS;
RHS = (tmp(1,:).^2 + tmp(2,:).^2) / 2;
RHS = RHS.';

% Remove z=0

pos = zgrid==zbar;
LHS = Q;
LHS(:,pos) = 1;
sol = LHS \ RHS;
q = sqrt(sol(pos)*2);

end