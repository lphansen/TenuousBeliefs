function dv = HJBODE(z,v,theta,param)

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

% Define Derivatives

v1 = v(1);
v2 = v(2);

% Solve beta, kappa, alphaz

mined = S2(z,v2,param);

% Return

dv = [ v2; ...
    2/norm(sigmaz)^2 * ( delta*v1-mined+1/(2*theta)*([0.01 v2]*sigma*sigma.'*[0.01;v2]) ) ];

end