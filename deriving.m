close all
clear all

%About x (or [1, 0, 0])
C_1 = @(t) [1, 0, 0;
              0, cos(t), -sin(t);
              0, sin(t), cos(t)];
%About y (or [0, 1, 0])
C_2 = @(t) [cos(t), 0, sin(t);
                0, 1, 0;
                -sin(t), 0, cos(t)];
%About z (or [0, 0, 1])
C_3 = @(t) [cos(t), -sin(t), 0;
              sin(t), cos(t), 0;
              0, 0, 1];
          
          
%Make matrices for printing
syms a;
syms b;
syms c;

fprintf('Transform about the x (or [1, 0, 0]) axis is: \n')
pretty(C_1(a))
fprintf('Transform about the y (or [0, 1, 0]) axis is: \n')
pretty(C_2(a))
fprintf('Transform about the z (or [0, 0, 1]) axis is: \n')
pretty(C_3(a))


fprintf('Hence the 3-2-1 transform is: \n');
zyx = C_3(a)*C_2(b)*C_1(c);
pretty(zyx);

fprintf('Diff zyx with respect to a\n');
pretty(diff(zyx, 'a'));
fprintf('Diff zyx with respect to b\n');
pretty(diff(zyx, 'b'));
fprintf('Diff zyx with respect to c\n');
pretty(diff(zyx, 'c'));

syms x
syms y
syms z
p = [x; y; z]
uvw = zyx*p;
pretty(uvw)

fprintf('Diff uvw wrt a\n');
pretty(diff(uvw, 'a'))
fprintf('Diff uvw wrt b\n');
pretty(diff(uvw, 'b'))
fprintf('Diff uvw wrt b\n');
pretty(diff(uvw, 'c'))


