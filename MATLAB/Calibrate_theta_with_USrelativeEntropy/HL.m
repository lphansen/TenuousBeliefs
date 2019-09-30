function [hl,pol,tort,s1,s2,negraw,posraw] = HL(theta,param,dv0,zl,zr,dvl,dvr,Dz,calHL)

[pol,negraw,posraw] = MatchODE(theta,zl,zr,dvl,dvr,param,Dz,dv0);
[tort,s1,s2] = Distortion(pol,theta,param);
if calHL 
    rho = ChernoffEntropy(tort(3:4,:),pol.x,param);
    hl = log(2)/rho;
else
    hl = [];
end

end
