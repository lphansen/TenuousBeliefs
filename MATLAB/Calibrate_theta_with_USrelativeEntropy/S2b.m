function [mined,s2] = S2b(z,dv,param)

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

% Define objective

function obj = ToMin(x)
    s1 = S1(x,z,param);
    if (imag(s1)~=0)
        obj = Inf;
    else
        obj = 0.01*(alphayhat+betahat*(z-zbar)+S1(x,z,param)) + dv*(alphazhat-kappahat*(z-zbar)+x);
    end
end

% Minimization

[s2,mined] = fminsearch(@ToMin,[0.01*z],optimset('TolX',1e-12,'TolFun',1e-12,'MaxIter',100000,'MaxFunEvals',100000));
%x = ga(@ToMin,2,optimset('TolX',1e-12,'TolFun',1e-12));

end