function entropy = DrawEllipsoidr(z,param,color)

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

% Grid for s1 and s2

r1 = linspace(-0.2,0.2,1000);
r2 = linspace(-0.2,0.2,1000);
entropy = zeros(length(r1),length(r2));

for y = 1:length(r2)
    for x = 1:length(r1)
        r = [r1(x);r2(y)];
        s = sigma * r;
        entropy(x,y) = 0.5*(a*s(1)^2+2*b*s(1)*s(2)+d*s(2)^2) + (rho1+rho2*(z-zbar))*(alphazhat-kappahat*(z-zbar)+s(2)) + norm(sigmaz)^2/2*rho2-q^2/2;
    end
end

contour(r2,r1,entropy,[0 0],'LineWidth',3,'LineColor',color);

end