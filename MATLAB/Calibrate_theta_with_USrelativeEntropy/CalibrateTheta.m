function diff = CalibrateTheta(theta,param,dv0,zl,zr,dvl,dvr,Dz,tgtUSq)
% Modified from the function HL

[pol,~,~] = MatchODE(theta,zl,zr,dvl,dvr,param,Dz,dv0);
[tort,~,~] = Distortion(pol,theta,param);
qUS = RelativeEntropyUS(tort(3:4,:),tort(1:2,:),pol.x,param); 
diff = qUS-tgtUSq;

end
