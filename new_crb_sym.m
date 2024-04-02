clearvars
close all
clc

%% define variables and likelihood function

syms y1 y2 y3 T fD v l eta a b c K sp

% loglike = (y - 2*pi*T*(fD+(v/l)*(cos(eta - a) - cos(eta))))^2/(2*sp);

loglike = (y1 - f1(fD, v, eta, T, l, a))^2 / (2*sp) + (y2 - fm(v, eta, T, l, b))^2 / (2*sp) + (y3 - fm(v, eta, T, l, c))^2 / (2*sp);

%% compute all the 1st and second derivatives

dLdfD = diff(loglike, fD);
dLdeta = diff(loglike, eta);
dLdv = diff(loglike, v);

dLdfD2 = diff(loglike, fD, 2);
dLdeta2 = diff(loglike, eta, 2);
dLdv2 = diff(loglike, v, 2);

dLdfDdeta = diff(dLdfD, eta);
dLdfDdv = diff(dLdfD, v);
dLdvdeta = diff(dLdv, eta);
% % 
% pretty(dLdfD2)
% pretty(dLdeta2)
% pretty(dLdv2)
% pretty(dLdfDdeta)
% pretty(dLdfDdv)
% pretty(dLdvdeta)

%% define Fisher information matrix

Ey1 = f1(fD, v, eta, T, l, a);
Ey2 = fm(v, eta, T, l, b);
Ey3 = fm(v, eta, T, l, c);

A = -dLdfD2;
B = -dLdfDdeta;
C = -dLdfDdv;

E = -subs(dLdeta2, [y1, y2, y3], [Ey1, Ey2, Ey3]);  % here y remains and we take expectation
D = -subs(dLdvdeta, [y1, y2, y3], [Ey1, Ey2, Ey3]); % here y remains and we take expectation
F = -dLdv2;


% pretty(A)
% pretty(E)
% pretty(F)
% pretty(B)
% pretty(C)
% pretty(D)

FI = [[A, B, C]; [B, E, D]; [C, D, F]];

invFI = inv(FI);


pretty(simplify(invFI(1,1)))


% pretty(simplify(invFI(2,2)))


% pretty(simplify(invFI(3,3)))

%%

fd2 = subs(invFI(1,1), [a,b,c], [5*pi/6, pi/4, pi/6]);


%pretty(simplify(fd2))





function y1 = f1(fD, v, eta, T, l, x)
y1 = 2*pi*T*(fD+(v/l)*(cos(eta - x) - cos(eta)));
end

function ym = fm(v, eta, T, l, x)
ym = 2*pi*T*(v/l)*(cos(eta - x) - cos(eta));
end

