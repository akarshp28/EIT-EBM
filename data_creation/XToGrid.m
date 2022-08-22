function [iR,jC] = XToGrid(x,y,N,marg)
    center = (N+1)/2;
    h = 2/(N-1-2*marg);
    iR = (y/h+center); iR = round(iR); if iR<=0, iR=1; end; if iR > N, iR=N; end
    jC = (x/h+center); jC = round(jC); if jC<=0, jC=1; end; if jC > N, jC=N; end
end
