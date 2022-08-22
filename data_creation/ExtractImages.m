function [h,center, radius,U,W,Sigma,XOmeg,YOmeg, Omega,Sigmax,Sigmay, dudn, sdudn, n_hat_x, n_hat_y] = ExtractImages(p,e,t,u,g,N,Globalsx,Globalsy)
    
    marg = 10;
    phiThresh = 0.0001;
    center = (N+1)/2;
    h = 2.0/(N-1-2*marg);
    radius = (N-1-2*marg)/2;
    
    fprintf('center=%.f h=%f radius=%.f\n',center,h,radius);
    
    x = linspace(-1-h*marg,1+h*marg,N);
    y = linspace(-1-h*marg,1+h*marg,N);
    U = tri2grid(p,t,u,x,y);
    
    Nedges = size(e,2);
    Npoints = size(p,2);
    W = zeros(N);
    gg = zeros(Npoints,1);
    for ii=1:Nedges
         pt_idx = e(1,ii);
         gg(pt_idx) = g(ii);
     end
     G = tri2grid(p,t,gg,x,y);
     G(isnan(G)) = 0;
    W(G~=0) = 1;
    
    Sigma = zeros(N,N);
    Sigmax = zeros(N,N);
    Sigmay = zeros(N,N);
    x = linspace(-1-h*marg,1+h*marg,N);
    y = linspace(-1-h*marg,1+h*marg,N);
    for ii=1:length(x)
        for jj=1:length(y)
            xx = x(ii); yy=y(jj);
            [i,j] = XToGrid(xx,yy,N,marg);
            
            Sigma(i,j) = MySigma(xx+1i*yy);
            [itmp,jtmp] = XToGrid(xx,yy,size(Globalsx,1),0);
            Sigmax(i,j) = Globalsx(itmp,jtmp);
            Sigmay(i,j) = Globalsy(itmp,jtmp);
        end
    end
    
    phi=zeros(N);
    for x=1:N
        for y=1:N
            phi(y,x) = radius-sqrt( (x-center).^2 + (y-center).^2 );
        end
    end    
    W(phi > -1*phiThresh & phi < phiThresh) = 1;
    
    [phi_x,phi_y] = gradient(phi);
    magphi = sqrt(phi_x.^2+phi_y.^2);
    n_hat_x = -phi_x./(magphi+1e-20); n_hat_y = -phi_y./(magphi+1e-20);
    
    IX = find(phi >=-phiThresh);
    [YOmeg,XOmeg] = ind2sub(size(U),IX);
    
    [ux,uy] = gradient(U,h);
    dudn = ux.*n_hat_x +uy.*n_hat_y;
    
    U(isnan(U)) = 0;
    dudn(isnan(dudn))=0;
    Sigma(isnan(Sigma)) = 0;
    sdudn = Sigma.*dudn;
    
    Omega = zeros(size(U));
    for i=1:length(XOmeg)
        xx = XOmeg(i); yy = YOmeg(i);
        Omega(yy,xx)=1;
    end
    
    d = bwdist(1-Omega);
    W = zeros(size(d));
    W(d>0 & d <3) = 1;
    W(U==0) = 0;
    Omega(U==0)=0;
end
