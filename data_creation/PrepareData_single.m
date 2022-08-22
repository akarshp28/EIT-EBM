clc
close all
close all

%% Data parameters
k = 1;                  % image number

% anomaly mode
% 0 = circle
% 1 = shepp-logan phantom
circle = 0;
anomaly_type = 4;

N = 128;                % mesh size
NumOfRefine = 7;        % mesh density
factor = 8;             % current factor
solve = 1;

if circle == 0
    
    if anomaly_type == 1
        sigma_val = 8;
        hc1 = 0.25;  hc2 = 0.0;
        he1 = 0.1;   he2 = 0.3;
        hR = 0.09;
        
        anomaly_list = [sigma_val, hc1, hc2, hR, he1, he2];
    
    elseif anomaly_type == 2
        
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
    
    elseif anomaly_type == 3
        
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
    
    anomaly_plot(N, anomaly_type, anomaly_list)
end

if solve > 0
    
    nlist = [1, 2];
    phslist = [0, 2, 4, 6];
    
    for nval = 1:length(nlist)
        n = nlist(nval);
        for pval = 1:length(phslist)
            phs = phslist(pval);
            create_data(k, circle, anomaly_type, anomaly_list, n, phs, NumOfRefine, N, factor)
        end
    end
end
