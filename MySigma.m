function a = MySigma(z)
    global GlobalSigma;
    N = size(GlobalSigma,1);
    x = real(z); y = imag(z);
    [i,j] = XToGrid(x,y,N,0);
    z1 = sub2ind([N,N],i,j);
    a = GlobalSigma(z1);    
end
