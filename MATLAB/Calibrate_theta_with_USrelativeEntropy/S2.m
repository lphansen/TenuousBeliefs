function [mined,s2] = S2(z,dv,param)

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

% Solve S2

A = 0.5*a;
C0 = (rho1+rho2*(z-zbar))*(alphazhat-kappahat*(z-zbar)) + norm(sigmaz)^2/2*rho2 - q^2/2;
C1 = (rho1+rho2*(z-zbar));
C2 = 0.5*d;
D = b^2/(2*A)-2*C2;
E = ( 100*dv-b/(2*A) )^2;

AA = E*b^2-4*A*E*C2-D^2;
BB = 2*C1*D-4*A*E*C1;
CC = -4*A*C0*E-C1^2;

s21 = ( -BB + sqrt(BB^2-4*AA*CC) ) / (2*AA);
s22 = ( -BB - sqrt(BB^2-4*AA*CC) ) / (2*AA);

% Minimization

mined1 = 0.01*(alphayhat+betahat*(z-zbar)+S1(s21,z,param)) + dv*(alphazhat-kappahat*(z-zbar)+s21);
mined2 = 0.01*(alphayhat+betahat*(z-zbar)+S1(s22,z,param)) + dv*(alphazhat-kappahat*(z-zbar)+s22);
mined = min([mined1,mined2]);
s2 = (mined1<=mined2)*s21 + (mined1>mined2)*s22;

end