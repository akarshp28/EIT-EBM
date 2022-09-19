clc
clear all
close all

% For paper
% Phantom 1 = circle = 1, logan_type = 1;
% Phantom 2 = circle = 0, circle_anomaly_type = 1;
% Phantom 3 = circle = 1, logan_type = 5;
% Phantom 4 = circle = 0, circle_anomaly_type = 2;
% Phantom 5 = circle = 0, circle_anomaly_type = 4;

phantom = 1;

if phantom == 1
    circle = 1;
    circle_anomaly_type = nan;
    logan_type = 1;

elseif phantom == 2
    circle = 0;
    circle_anomaly_type = 1;
    logan_type = nan;

elseif phantom == 3
    circle = 1;
    circle_anomaly_type = nan;
    logan_type = 5;

elseif phantom == 4
    circle = 0;
    circle_anomaly_type = 2;
    logan_type = nan;
    
elseif phantom == 5
    circle = 0;
    circle_anomaly_type = 4;
    logan_type = nan;
end

% final mesh data with anomaly
[p,e,t, sigma] = generate_phantom(circle, logan_type, circle_anomaly_type);

pdemesh(p,e,t, sigma)

% function to generate mesh type data
function [p, e, t, cond] = generate_phantom(circle, logan_type, circle_anomaly_type)
    
    NumOfRefine = 7;
    
    g = 'circleg';
    [p,e,t]=initmesh(g);
    for i=1:NumOfRefine
        [p,e,t]=refinemesh(g,p,e,t);
    end
    
    % Centers of mass of triangles
    xvec = p(1,:);
    trix = mean(xvec(t(1:3,:)));
    yvec = p(2,:);
    triy = mean(yvec(t(1:3,:)));

    if circle == 0
        
        % Evaluate conductivity Circle based phantoms

        if circle_anomaly_type == 1
            sigma_val = 8;
            hc1 = 0.25;  hc2 = 0;
            he1 = 0.1;   he2 = 0.3;
            hR = 0.18;
            
            anomaly_list = [sigma_val, hc1, hc2, hR, he1, he2];
        
        elseif circle_anomaly_type == 2
            
            %anomaly 1
            sigma_val1 = 8;
            hc1 = 0;            hc2 = -0.25;
            he1 = 1;            he2 = 1;
            hR = 0.2;
            
            % anomaly 2
            sigma_val2 = 5;
            hc3 = 0;            hc4 = 0.5;
            he3 = 1;            he4 = 1;
            
            anomaly_list = [sigma_val1, hc1, hc2, hR, he1, he2, ...
                            sigma_val2, hc3, hc4, hR, he3, he4];
        
        elseif circle_anomaly_type == 3
            
            % anomaly 1
            sigma_val1 = 10;
            hc1 = 0.45;        hc2 = -0.4;
            he1 = 1;        he2 = 1;
            hR = 0.2;
            
            % anomaly 2
            sigma_val2 = 8;
            hc3 = 0;        hc4 = 0;
            he3 = 1;        he4 = 1;
            
            % anomaly 3
            sigma_val3 = 5;
            hc5 = -0.45;        hc6 = 0.4;
            he5 = 1;        he6 = 1;
            
            anomaly_list = [sigma_val1, hc1, hc2, hR, he1, he2, ...
                            sigma_val2, hc3, hc4, hR, he3, he4, ...
                            sigma_val3, hc5, hc6, hR, he5, he6];
        
        else
            
            % anomaly 1
            sigma_val1 = 10;
            hc1 = 0.45;        hc2 = -0.4;
            he1 = 1;        he2 = 1;
            hR = 0.2;
            
            % anomaly 2
            sigma_val2 = 8;
            hc3 = 0;        hc4 = 0;
            he3 = 1;        he4 = 1;
            
            % anomaly 3
            sigma_val3 = 5;
            hc5 = -0.45;        hc6 = 0.4;
            he5 = 1;            he6 = 1;
            
            % anomaly 4
            sigma_val4 = 15;
            hc7 = 0.2;        hc8 = 0.6;
            he7 = 1;            he8 = 1;
            
            anomaly_list = [sigma_val1, hc1, hc2, hR, he1, he2, ...
                            sigma_val2, hc3, hc4, hR, he3, he4, ...
                            sigma_val3, hc5, hc6, hR, he5, he6, ...
                            sigma_val4, hc7, hc8, hR, he7, he8, ...
                            ];
        
        end
        
        anomaly_plot(128, circle_anomaly_type, anomaly_list)
        
        cond = anomaly_gen(trix+1i*triy, circle_anomaly_type, anomaly_list);
    
    else
        
        % Evaluate conductivity Shepp-Logan Phantom
        cond = heartNlungs(trix + 1i*triy, logan_type);
    
    end

end