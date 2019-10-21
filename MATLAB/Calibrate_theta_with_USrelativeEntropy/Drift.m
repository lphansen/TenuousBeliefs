function out = Drift(out,param)

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

% Drift

drift = sigma * out.rh(3:4,:);
out.drifty = drift(1,:) + alphayhat + betahat*(out.v.x-zbar);
out.driftz = drift(2,:) + alphazhat - kappahat*(out.v.x-zbar);

% 2nd derivative

d2v = zeros(1,length(out.v.x));
parfor j = 1:length(out.v.x)
    temp = HJBODE(out.v.x(j),out.v.y(:,j),out.theta,param);
    d2v(j) = temp(2);
end
%d2v(abs(out.v.x)<1e-6) = NaN;
if (size(out.v.y,1)>2)
    out.v.y(3,:) = d2v;
else
    out.v.y = [out.v.y;d2v];
end

end
