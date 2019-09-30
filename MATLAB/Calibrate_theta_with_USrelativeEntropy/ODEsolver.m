function sol = ODEsolver(zrange,bdl,bdr,theta,param)

tosolve = @(z,v) HJBODE(z,v,theta,param);
bc = @(dvl,dvr) [dvl(2)-bdl;dvr(2)-bdr];
if abs(zrange(1)) >= abs(zrange(2))
    temp = bdr;
else
    temp = bdl;
end
solintl = bvpinit(linspace(zrange(1),zrange(2),10),[0 temp]);
sol = bvp4c(tosolve,bc,solintl);

end