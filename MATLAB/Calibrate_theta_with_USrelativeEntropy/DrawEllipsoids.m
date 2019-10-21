function entropy = DrawEllipsoids(z,param,color)

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

s1 = linspace(-0.1,0.1,800);
s2 = linspace(-0.01,0.01,1000);
entropy = zeros(length(s1),length(s2));

for y = 1:length(s2)
    for x = 1:length(s1)
        entropy(x,y) = 0.5*(a*s1(x)^2+2*b*s1(x)*s2(y)+d*s2(y)^2) + (rho1+rho2*(z-zbar))*(-kappahat*(z-zbar)+s2(y)) + norm(sigmaz)^2/2*rho2-q^2/2;
    end
end

contour(s2,s1,entropy,[0 0],'LineWidth',3,'LineColor',color);

end