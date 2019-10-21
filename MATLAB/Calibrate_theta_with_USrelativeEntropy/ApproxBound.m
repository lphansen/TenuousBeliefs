function sol = ApproxBound(param)

% Read param

q = param.q;

alphayhat = param.alphayhat;
alphazhat = param.alphazhat;
betahat = param.betahat;
kappahat = param.kappahat;
sigmay = param.sigmay;
sigmaz = param.sigmaz;
delta = param.delta;

rho1 = param.rho1;
rho2 = param.rho2;
zbar = param.zbar;
sigma = param.sigma;
a = param.a;
b = param.b;
d = param.d;

% Solve

%options = optimoptions('fsolve','MaxFunEvals',10000,'MaxIter',10000);
%sol = fsolve(@tosolve,[1 betahat kappahat],options);

syms nu s1tmp s2tmp;
sol = vpasolve(...
    [(-delta-kappahat+s2tmp)*nu + 0.01*(betahat+s1tmp)==0,...
    nu*(a*s1tmp+b*s2tmp)-0.01*(b*s1tmp+d*s2tmp+rho2)==0, ...
    0.5*(a*s1tmp^2+2*b*s1tmp*s2tmp+d*s2tmp^2)+rho2*(-kappahat+s2tmp) == 0], ...
    [nu s1tmp s2tmp]);

end