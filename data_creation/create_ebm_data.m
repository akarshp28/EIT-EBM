function create_ebm_data(k, circle, anomaly_type, anomaly_list, NumOfRefine, N)

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

fil = fspecial('gaussian',200, 3);
SigmaSmooth = imfilter(GlobalSigma,fil,'symmetric','same');
GlobalSigma = SigmaSmooth;

%% Solving the equation by the finite element method
% n = 1;
% phs = 0;
% factor = 8;
% phase = phs*pi/8;
% save ./Data/InputData/BoundaryDataN n phase factor;
u = assempde('BoundaryData',p,e,t,'FEMconductivitySmooth',0,0);
[~,g,~,~] = BoundaryData(p,e);

%% Represent the finite element method solution to caresian grid
[h,center, radius, W,Sigma,U,XOmeg,YOmeg, Yall,Xall, Omega] = ExtractImages_ebm(p,e,t,u,g,N);
CorrectS = (0.5*(max(XOmeg)-min(XOmeg)));
CorrectB = (min(XOmeg)+CorrectS);

imagesc(Sigma .* Omega);
axis equal;
colorbar;
title('Final Sigma');
f = gcf;

%% Save data
currentFolder = pwd;
tmpl = ['phantom_', num2str(k)];
pt = sprintf('%s\\Data\\InputData\\',currentFolder);
file = sprintf('%s_N%d_refine%d.mat', tmpl, N, NumOfRefine);

fprintf('saving %s\n',fullfile(pt,file));

save(fullfile(pt,file), 'Omega', 'W', 'XOmeg','YOmeg', 'Yall', 'Xall', 'h','center', 'radius', ...
    'CorrectS','CorrectB', 'Sigma', 'U', 'anomaly_list');

file = sprintf('%s_N%d_refine%d.png', tmpl, N, NumOfRefine);
exportgraphics(f,fullfile(pt,file),'Resolution',300)

end
