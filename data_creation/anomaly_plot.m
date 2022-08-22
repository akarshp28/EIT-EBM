function anomaly_plot(N, anomaly_type, anomaly_list)
    
    % Create evaluation points
    t       = linspace(-1,1,N);
    [x1,x2] = meshgrid(t);
    z       = x1 + 1i*x2;
    
    % Evaluate potential    
    c = anomaly_gen(z, anomaly_type, anomaly_list);
    c(abs(z)>1) = 0;
    
    % Two-dimensional plot 
    figure
    imagesc(c)
    colormap jet
    map = colormap;
    colormap([[1 1 1];map]);
    axis equal
    axis off
    colorbar
    drawnow
end
