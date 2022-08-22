% Evaluates discontinuous heart-and-lungs phantom conductivity for complex 
% arguments. The heart and lungs are modelled as ellipses and
% the background conductivity is 1.
%
% Arguments:
% z   planar evaluation points, given as complex numbers.
%
% Samuli Siltanen May 2012

function ans = heartNlungs(z)

% Conductivities of heart and lung (background is 1)
back = 1;
lung = 5;
heart = 2;
  
% Initialize
[zrow,zcol] = size(z);
z           = z(:);
ans         = back*ones(size(z));
x1          = real(z);
x2          = imag(z);

% Build coarse representation of heart. Planar point (hc1,hc2) is the center of the ellipse
% describing the heart; numbers he1 and he2 give the eccentrities with respect to radius hR.
hc1 = -.1;
hc2 = .4;
he1 = .8;
he2 = 1;
hR  = .2;

% Compute elliptical "distance" of the evaluation points from heart
hd  = sqrt(he1*(x1-hc1).^2 + he2*(x2-hc2).^2);

% Set value of conductivity inside the heart 
ans(hd <= hR) = heart;

% Build coarse representation of two lungs
l1c1  = .5;
l1c2  = 0;
l1e1  = 3;
l1e2  = 1;
l1R   = .5;
fii   = -pi/7;
rot11 = cos(fii);
rot12 = sin(fii);
rot21 = -sin(fii);
rot22 = cos(fii);
l1d   = sqrt(l1e1*((rot11*x1+rot12*x2)-l1c1).^2 + l1e2*((rot21*x1+rot22*x2)-l1c2).^2);

ans(l1d <= l1R) = lung;

l2c1 = -.6;
l2c2 = 0;
l2e1 = 3;
l2e2 = 1;
l2R  = .4;
fii   = pi/7;
rot11 = cos(fii);
rot12 = sin(fii);
rot21 = -sin(fii);
rot22 = cos(fii);
l2d  = sqrt(l2e1*((rot11*x1+rot12*x2)-l2c1).^2 + l2e2*((rot21*x1+rot22*x2)-l2c2).^2);
ans(l2d <= l2R) = lung;

% Reshape answer to original form
ans = reshape(ans,[zrow,zcol]);

