function [rh,s1,s2] = Distortion(sol,theta,param)

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

% Calculate R

Nz = size(sol.x,2);
s2 = zeros(1,Nz);
s1 = zeros(1,Nz);

for j = 1:Nz
    [~,s2(j)] = S2(sol.x(j),sol.y(2,j),param);
    s1(j) = S1(s2(j),sol.x(j),param);
end

s = [s1;s2];
r = sigma \ s;

% Calculate H

h = -1/theta * sigma.'*[0.01*ones(1,Nz);sol.y(2,:)] + r;

% Output

rh = [r;h];

end