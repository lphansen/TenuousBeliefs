function sol=FeynmanKac(muz,sigmaz,zgrid,fintl,T,Dt)
% This function solves one dimension Feynman-Kac equation with initial
% condition using implicit FD, boundaries should be natural.

Dz = zgrid(2) - zgrid(1);

A = sparse(- eye(length(zgrid)));

for j = 2 : length(zgrid)-1
    A(j,j) = A(j,j) + Dt*( -norm(sigmaz)^2/(Dz^2) );
    A(j,j+1) = Dt * ( muz(j)/(2*Dz) + 0.5*norm(sigmaz)^2/(Dz^2) );
    A(j,j-1) = Dt * ( -muz(j)/(2*Dz) + 0.5*norm(sigmaz)^2/(Dz^2) );
end

a1 = A(2,1);
a2 = A((end-1),end);
A = A(2:(end-1),2:(end-1));

phiold = fintl;
sol = zeros(length(zgrid),T/Dt+1);
sol(:,1) = fintl;

for t = 1:(T/Dt)
    b = -phiold(2:(end-1));
    b(1) = b(1) - a1*phiold(1);
    b(end) = b(end) - a2*phiold(end);
    phinew = A\b;
    phinew1 = 2*phinew(1)-phinew(2);
    phinewN = 2*phinew(end) - phinew(end-1);
    phinew = [phinew1; phinew; phinewN];
    phiold = phinew;
    
    sol(:,t+1) = phinew;
end