function create_data(k, circle, anomaly_type, anomaly_list, n, phs, NumOfRefine, N, factor)

fprintf('n=%d Phase=%d \n', n, phs);

%% Preparing mesh
g = 'circleg';
[p,e,t]=initmesh(g);
for i=1:NumOfRefine
    [p,e,t]=refinemesh(g,p,e,t);
end

%% Arrange Sigma as GlobNxGlobN matrix
global GlobalSigma;
xx = linspace(-1,1,N);
yy = linspace(-1,1,N);
GlobalSigma = zeros(N,N);

if circle == 0
    % Circle based anomaly
    for ii=1:N
        for jj=1:N
            GlobalSigma(jj,ii) = anomaly_gen(xx(ii)+1i*yy(jj), anomaly_type, anomaly_list);
        end
    end
else
    % Shepp-Logan Phantom
    for ii=1:N
        for jj=1:N
            GlobalSigma(jj,ii) = heartNlungs(xx(ii)+1i*yy(jj));
        end
    end
end

noise_std = 3;
fil = fspecial('gaussian',200, noise_std); %noise std
SigmaSmooth = imfilter(GlobalSigma,fil,'symmetric','same');
GlobalSigma = SigmaSmooth;

[Globalsx,Globalsy] = gradient(GlobalSigma);

%% Solving the equation by the finite element method
phase = phs*pi/8;
save ./Data/InputData/BoundaryDataN n phase factor;
u = assempde('BoundaryData',p,e,t,'FEMconductivitySmooth',0,0);
[~,g,~,~] = BoundaryData(p,e);

%% Represent the finite element method solution to caresian grid
[h,center, radius, U,W,Sigma,XOmeg,YOmeg,Omega,sx,sy, dudn, sdudn, n_hat_x, n_hat_y] = ExtractImages(p,e,t,u,g,N,Globalsx,Globalsy);
CorrectS = (0.5*(max(XOmeg)-min(XOmeg)));
CorrectB = (min(XOmeg)+CorrectS);
U = U.*Omega;

%% Current data
Xnor = (XOmeg-CorrectB)/CorrectS;
Ynor = (YOmeg-CorrectB)/CorrectS;
g = zeros(N); %the current
for ii=1:length(Xnor)
    xx = XOmeg(ii); yy = YOmeg(ii);
    theta = angle(Xnor(ii)+1i*Ynor(ii));
    g(yy,xx) = real( factor* (1/sqrt(2*pi)*exp(1i*n*(theta+phase))));
end
G = g.*W;

%% Preparing the gradients, with a special care on the boundary
[ux,uy] = BoundaryGradient_ver1(U,W,Omega,CorrectB,CorrectS);
[uxx,~] = BoundaryGradient_ver1(ux,W,Omega,CorrectB,CorrectS);
[~,uyy] = BoundaryGradient_ver1(uy,W,Omega,CorrectB,CorrectS);

%% Domain data
dom = Omega.*(1-W);
ixd = find(dom>0);
[Yd,Xd] = ind2sub(size(dom),ixd);
sd = Sigma(ixd);
sxd = sx(ixd);
syd = sy(ixd);
ud = U(ixd);
uxd = ux(ixd);
uyd = uy(ixd);
uxxd = uxx(ixd);
uyyd = uyy(ixd);

%% Boundary data
ix2 = find(W>0);
[Yb,Xb] = ind2sub(size(W),ix2);
ub = U(ix2);
sb = Sigma(ix2);
sxb = sx(ix2);
syb = sy(ix2);
uxb = ux(ix2);
uyb = uy(ix2);
uxxb = uxx(ix2);
uyyb = uyy(ix2);
gb = G(ix2);
n_hat_xb = n_hat_x(ix2);
n_hat_yb = n_hat_y(ix2);
dudnb = dudn(ix2);
sdudnb = sdudn(ix2);

%% Norm of PDE Check = 0 when alpha = 0
alphalist = linspace(-2,2,100);
divlist = zeros(1,length(alphalist));
for al = 1:length(alphalist)
    alpha = alphalist(al);
    div = alpha * sx * ux + alpha * sy * uy + alpha * (Sigma - 1) * (uxx + uyy);
    divlist(al) =  norm(div)^2;
end

figure
plot(alphalist, divlist)
xlabel('alpha')
ylabel('F(alpha)')
title(num2str(NumOfRefine), num2str(N))
drawnow

%% Norm check PDE
div = sx * ux + sy * uy + Sigma * (uxx + uyy);
fprintf('Div=%f \n', norm(div)^2);

%% Draw figures
draw_fig

%% Save data
currentFolder = pwd;
tmpl = ['phantom_', num2str(k)];
pt = sprintf('%s\\Data\\InputData\\',currentFolder);
file = sprintf('%s_N%d_refine%d_n%d_phs%d.mat', tmpl, N, NumOfRefine, n, phs);

fprintf('saving %s\n',fullfile(pt,file));

save(fullfile(pt,file), 'Omega', 'W', 'XOmeg','YOmeg', ...
                'h','center', 'radius','Xb','Yb', 'CorrectS','CorrectB', ...
                'dudn', 'sdudn', 'n_hat_x', 'n_hat_y', ...
                'dudnb', 'sdudnb', 'n_hat_xb', 'n_hat_yb',...
            'Sigma','sx', 'sy', 'U','ux', 'uy', 'uxx', 'uyy', ...
            'Xd', 'Yd', 'sd', 'sxd', 'syd', 'ud', 'uxd', 'uyd', 'uxxd', 'uyyd', ...
            'uxb', 'uyb', 'uxxb', 'uyyb', 'sb', 'sxb', 'syb', 'ub','gb', ...
            'n', 'phs', 'anomaly_type', 'anomaly_list', 'G');

end
