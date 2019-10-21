function [exp,density] = StationaryExpect(tort,integrand,zgrid,param)

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

% Drift distortion

drift = sigma * tort;
muz = drift(2,:) + alphazhat - kappahat*zgrid;

dz = zgrid(2) - zgrid(1);
nz = size(zgrid,2);

%sol = FeynmanKac(muz,sigmaz,zgrid,integrand,10000,0.1);
%exp = mean(sol(abs(zgrid)<0.5,end));

sol = KolmogorovF(muz,sigmaz,zgrid,ones(nz,1),20000,0.1);
density = (sol(:,end));
if ( min(density)< -1e-10 )
    error('Density not found');
end    
density = abs(density) ./ (sum(abs(density).*dz));
exp = sum(integrand.*density.*dz);

end
