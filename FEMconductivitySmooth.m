function cond = FEMconductivitySmooth(p,t,u,time)

% Centers of mass of triangles
xvec = p(1,:);
trix = mean(xvec(t(1:3,:))); 
yvec = p(2,:);
triy = mean(yvec(t(1:3,:))); 

% Evaluate conductivity with auxiliary routine
cond = MySigma(trix+1i*triy);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

