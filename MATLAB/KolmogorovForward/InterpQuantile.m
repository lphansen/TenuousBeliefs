function qt = InterpQuantile(zgrid,mgrid,z0)

dim = size(mgrid);
qt = [];
for t = 1:dim(2)
    tmp = interp1(zgrid,mgrid(:,t),z0);
    qt = [qt tmp];
end    
    