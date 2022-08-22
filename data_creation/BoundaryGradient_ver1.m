function [ux,uy] = BoundaryGradient_ver1(U, W, Omega, CorrectB, CorrectS)
   
    Unan = U;
    Unan(Omega==0) = nan;
    [uxtmp,uytmp]=gradient(Unan);
    ux = uxtmp;
    ux(W==1)=0;ux(isnan(ux))=0;
    uy = uytmp;
    uy(W==1)=0;uy(isnan(uy))=0;
    

    idx = find(W==1);
    [Yb,Xb] = ind2sub(size(W),idx);
    
    Xnor = (Xb-CorrectB)/CorrectS;
    Ynor = (Yb-CorrectB)/CorrectS;
    
    idx = find(Xnor <0);
    for i=1:length(idx)
        x = Xb(idx(i)); y = Yb(idx(i));
        val = 0.5*(4*U(y,x+1)-3*U(y,x)-U(y,x+2));
        ux(y,x) = val;
    end
    
    idx = find(Xnor > 0);
    for i=1:length(idx)
        x = Xb(idx(i)); y = Yb(idx(i));
        val = 0.5*(-4*U(y,x-1)+3*U(y,x)+U(y,x-2));
        ux(y,x) = val;
    end
    
    
    idx = find(Ynor <0);
    for i=1:length(idx)
        x = Xb(idx(i)); y = Yb(idx(i));
        val = 0.5*(4*U(y+1,x)-3*U(y,x)-U(y+2,x));
        uy(y,x) = val;
    end
    
    idx = find(Ynor > 0);
    for i=1:length(idx)
        x = Xb(idx(i)); y = Yb(idx(i));
        val = 0.5*(-4*U(y-1,x)+3*U(y,x)+U(y-2,x));
        uy(y,x) = val;
    end
    
    
%     figure;
%     subplot(121);
%     imagesc(ux); title('ux');axis equal;
%     subplot(122);
%     imagesc(uy); title('uy'); axis equal;
