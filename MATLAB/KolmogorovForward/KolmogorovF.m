function sol=KolmogorovF(muz,sigmaz,zgrid,fintl,T,Dt)
% This function solves one dimension Kolmogorov Forward equation with initial
% condition using implicit upwind FD.

Dz = zgrid(2) - zgrid(1);

A = sparse(- eye(length(zgrid)));

for j = 2 : length(zgrid)-1
%     A(j,j) = A(j,j) + Dt*( -norm(sigmaz)^2/(Dz^2) );
%     A(j,j+1) = Dt * ( -muz(j+1)/(2*Dz) + 0.5*norm(sigmaz)^2/(Dz^2) );
%     A(j,j-1) = Dt * ( muz(j-1)/(2*Dz) + 0.5*norm(sigmaz)^2/(Dz^2) );
    A(j,j) = A(j,j) + Dt*( muz(j)/Dz*(muz(j)<0)-muz(j)/Dz*(muz(j)>0) -norm(sigmaz)^2/(Dz^2) );
    A(j,j+1) = Dt * ( -muz(j+1)/Dz*(muz(j)<0) + 0.5*norm(sigmaz)^2/(Dz^2) );
    A(j,j-1) = Dt * ( muz(j-1)/Dz*(muz(j)>0) + 0.5*norm(sigmaz)^2/(Dz^2) );
end

a1 = A(2,1);
a2 = A((end-1),end);
A = A(2:(end-1),2:(end-1));

phiold = fintl;
sol = zeros(length(zgrid),T/Dt+1);
sol(:,1) = fintl;

for t = 1:(T/Dt)
    b = -phiold(2:(end-1));
    b(1) = b(1) - a1*0;
    b(end) = b(end) - a2*0;
    phinew = A\b;
    phinew1 = 0;
    phinewN = 0;
    phinew = [phinew1; phinew; phinewN];
    phiold = phinew;
    
    sol(:,t+1) = phinew;
end