figure
scatter(XOmeg, YOmeg)
hold on
scatter(Xb, Yb)
hold off
drawnow

figure
subplot(121); imagesc(Sigma .* Omega); axis equal; colorbar; title('Sigma');
subplot(122); imagesc(U .* Omega); axis equal; colorbar; title('U');
drawnow

figure
subplot(121); imagesc(sx.* Omega); axis equal; colorbar; title('Sx');
subplot(122); imagesc(sy.* Omega); axis equal; colorbar; title('Sy');
drawnow

figure
subplot(121); imagesc(ux.* Omega); axis equal; colorbar; title('Ux');
subplot(122); imagesc(uy.* Omega); axis equal; colorbar; title('Uy');
drawnow

figure
subplot(121); imagesc(uxx.* Omega); axis equal; colorbar; title('Uxx');
subplot(122); imagesc(uyy.* Omega); axis equal; colorbar; title('Uyy');
drawnow