clc
close all
close all


for k=1:1000
    
    range = 0.7;
    single_Exp = 0;
    
    %k = number_gen(0, 1000, 1);
    disp(k)
    
    % anomaly mode
    % 0 = generate circle
    % 1 = shepp-logan phantom
    circle = 0;
    
    N = 128;                 % mesh size
    NumOfRefine = 7;        % mesh density
    
    anomaly_type = number_gen(1, 3, 1);
    disp(anomaly_type)
    
    ttt     = linspace(-1,1,N);
    [x1,x2] = meshgrid(ttt);
    z       = x1 + 1i*x2;
    
    for kk = 1:100000
        
        if anomaly_type == 1
            
            % xy-axis pos anomaly 1
            hc1 = number_gen(-range, range, 2);
            hc2 = number_gen(-range, range, 2);
            % sigma
            sigma_val1 = number_gen(3, 15, 1);
            % shape
            he1 = number_gen(0.3, 1, 2);
            he2 = number_gen(0.3, 1, 2);
            hR = number_gen(0.1, 0.2, 2);
            
            if sqrt((hc1)^2 + (hc2)^2) <= 0.7
                if sqrt(hR*(hc1)^2 + hR*(hc2)^2) <= 0.5 && sqrt(hR*he1*(hc1)^2 + hR*he2*(hc2)^2) <= 0.5
                    anomaly_list = [sigma_val1, hc1, hc2, hR, he1, he2];
                    break
                end
            end
        
        elseif anomaly_type == 2
            
            % xy-axis pos anomaly 1
            hc1 = number_gen(-range, range, 2);
            hc2 = number_gen(-range, range, 2);
            % sigma
            sigma_val1 = number_gen(3, 15, 1);
            % shape
            he1 = number_gen(0.5, 1, 2);
            he2 = number_gen(0.5, 1, 2);
            hR = number_gen(0.05, 0.25, 2);
            
            % xy-axis pos anomaly 2
            hc3 = number_gen(-range, range, 2);
            hc4 = number_gen(-range, range, 2);
            % sigma
            sigma_val2 = number_gen(3, 15, 1);
            % shape
            he3 = number_gen(0.5, 1, 2);
            he4 = number_gen(0.5, 1, 2);
            
            if sqrt((hc1)^2 + (hc2)^2) <= 0.65 && sqrt((hc3)^2 + (hc4)^2) <= 0.65
                if sqrt((hc1 - hc3)^2 + (hc2 - hc4)^2) >= 0.25
                    if sqrt(hR*(hc1 - hc3)^2 + hR*(hc2 - hc4)^2) >= 0.25
                        if sqrt(hR*he1*(hc1 - hc3)^2 + hR*he2*(hc2 - hc4)^2) >= 0.25
                            if sqrt(hR*he3*(hc1 - hc3)^2 + hR*he4*(hc2 - hc4)^2) >= 0.25
                                anomaly_list = [sigma_val1, hc1, hc2, hR, he1, he2, ...
                                                sigma_val2, hc3, hc4, hR, he3, he4];
                                break
                            end
                        end
                    end
                end
            end
        
        else

            % xy-axis pos anomaly 1
            hc1 = number_gen(-range, range, 2);
            hc2 = number_gen(-range, range, 2);
            % sigma
            sigma_val1 = number_gen(3, 15, 1);
            % shape
            he1 = number_gen(0.4, 1, 2);
            he2 = number_gen(0.4, 1, 2);
            hR = number_gen(0.05, 0.2, 2);
            
            % xy-axis pos anomaly 2
            hc3 = number_gen(-range, range, 2);
            hc4 = number_gen(-range, range, 2);
            % sigma
            sigma_val2 = number_gen(3, 15, 1);
            % shape 
            he3 = number_gen(0.4, 1, 2);
            he4 = number_gen(0.4, 1, 2);
            
            % xy-axis pos anomaly 3
            hc5 = number_gen(-range, range, 2);
            hc6 = number_gen(-range, range, 2);
            % sigma
            sigma_val3 = number_gen(3, 15, 1);
            % shape
            he5 = number_gen(0.5, 1, 2);
            he6 = number_gen(0.5, 1, 2);
            
            if sqrt((hc1)^2 + (hc2)^2) <= 0.6 && sqrt((hc3)^2 + (hc4)^2) <= 0.6 && sqrt((hc5)^2 + (hc6)^2) <= 0.6
                if sqrt((hc1 - hc3)^2 + (hc2 - hc4)^2) >= 0.3 && sqrt((hc1 - hc5)^2 + (hc2 - hc6)^2) >= 0.3
                    if sqrt((hc3 - hc5)^2 + (hc4 - hc6)^2) >= 0.3
                        if sqrt(hR*(hc1 - hc3)^2 + hR*(hc2 - hc4)^2) >= 0.3 && sqrt(hR*(hc1 - hc5)^2 + hR*(hc2 - hc6)^2) >= 0.3
                            if sqrt(hR*(hc3 - hc5)^2 + hR*(hc4 - hc6)^2) >= 0.3
                                anomaly_list = [sigma_val1, hc1, hc2, hR, he1, he2, ...
                                                sigma_val2, hc3, hc4, hR, he3, he4, ...
                                                sigma_val3, hc5, hc6, hR, he5, he6];
                                break
                            end
                        end
                    end
                end
            end
        end
    end
    
    disp(anomaly_list)

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
    
    promptMessage = sprintf('Do you want to Continue processing,\nor Cancel to abort processing?');
    button = questdlg(promptMessage, 'Continue', 'Continue', 'Terminate', 'Continue');
    
    if strcmpi(button, 'Terminate')
        close all;
        clear all;
        clc;
        
    else
        
        if single_Exp > 0
            
            create_ebm_data(k, circle, anomaly_type, anomaly_list, NumOfRefine, N)
            
        else
            
            nlist = [1, 2];
            phslist = [0, 2, 4, 6];
            factor = 8;
            
            for nval = 1:length(nlist)
                n = nlist(nval);
                for pval = 1:length(phslist)
                    phs = phslist(pval);
                    create_data(k, circle, anomaly_type, anomaly_list, n, phs, NumOfRefine, N, factor)
                end
            end

        end
    
    end

end

