function [polish,negsol,possol] = MatchODE(theta,zl,zr,dvl,dvr,param,Dz,dv0guess)

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

% Match v0

function diff = v0Diff(dv0)
    disp(['trying v''(0) = ',num2str(dv0)]);
    negsol = ODEsolver([zl 0],dvl,dv0,theta,param);
    disp(['For this case, v(0-) = ',num2str(negsol.y(1,end))]);
    possol = ODEsolver([0 zr],dv0,dvr,theta,param);
    disp(['For this case, v(0+) = ',num2str(possol.y(1,1))]);
    diff = negsol.y(1,end) - possol.y(1,1);
    disp(['The difference is ',num2str(diff)]);
end
        
dv0 = fzero(@v0Diff,dv0guess);
disp(['---------------------------']);
disp(['dv matched at value ',num2str(dv0)]);

% Negative parts

negsol = ODEsolver([zl 0],dvl,dv0,theta,param);
v1 = negsol.y(1,end);
v2 = negsol.y(2,end);
mined = S2(-1e-6,v2,param);
v3 = 2/norm(sigmaz)^2 * ( delta*v1-mined+1/(2*theta)*([0.01 v2]*sigma*sigma.'*[0.01;v2]) );
disp(['For theta ',num2str(theta),' v(0-) = ',num2str(v1)]);
disp(['For theta ',num2str(theta),' v''(0-) = ',num2str(v2)]);
disp(['For theta ',num2str(theta),' v"(0-) = ',num2str(v3)]);

% Positive parts

possol = ODEsolver([0 zr],dv0,dvr,theta,param);
v1 = possol.y(1,1);
v2 = possol.y(2,1);
mined = S2(1e-6,v2,param);
v3 = 2/norm(sigmaz)^2 * ( delta*v1-mined+1/(2*theta)*([0.01 v2]*sigma*sigma.'*[0.01;v2]) );
disp(['For theta ',num2str(theta),' v(0+) = ',num2str(v1)]);
disp(['For theta ',num2str(theta),' v''(0+) = ',num2str(v2)]);
disp(['For theta ',num2str(theta),' v"(0+) = ',num2str(v3)]);

% Polish solution

if (Dz==0)
    polish = [];
    return
end

% tosolve = @(z,v) HJBODE(z,v,theta,param);
% [negpol.x,negpol.y] = ode45(tosolve,zl:Dz:0,negsol.y(:,1));
% disp(['Polished results: v(0-) = ',num2str(negpol.y(end,1))]);
% disp(['Polished results: v''(0-) = ',num2str(negpol.y(end,2))]);
% [pospol.x,pospol.y] = ode45(tosolve,zr:(-Dz):0,possol.y(:,end));
% disp(['Polished results: v(0+) = ',num2str(pospol.y(end,1))]);
% disp(['Polished results: v''(0+) = ',num2str(pospol.y(end,2))]);
% 
% polish.x = [negpol.x; pospol.x((end-1):-1:1)].';
% polish.y = [negpol.y; pospol.y((end-1):-1:1,:)].';

% temp1 = spline(negsol.x,negsol.y(1,:),zl:Dz:0);
% temp2 = spline(negsol.x,negsol.y(2,:),zl:Dz:0);
% temp3 = spline(possol.x,possol.y(1,:),0:Dz:zr);
% temp4 = spline(possol.x,possol.y(2,:),0:Dz:zr);

temp1 = deval(negsol,zl:Dz:0);
temp2 = deval(possol,0:Dz:zr);

polish.x = zl:Dz:zr;
polish.y = [temp1 temp2(:,2:end)];

end