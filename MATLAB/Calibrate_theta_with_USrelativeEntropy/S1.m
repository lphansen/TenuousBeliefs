function s1 = S1(s2,z,param)

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

% Calculate aux

A = 0.5*a;
B = b*s2;
C = 0.5*d*s2^2 + (rho1+rho2*(z-zbar))*s2 + (rho1+rho2*(z-zbar))*(alphazhat-kappahat*(z-zbar)) + norm(sigmaz)^2/2*rho2 - q^2/2;

s1 = (-B - sqrt(B^2-4*A*C)) / (2*A);

end
