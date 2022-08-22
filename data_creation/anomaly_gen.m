function final_sigma = anomaly_gen(z, anomaly_type, anomaly_list)

if anomaly_type == 1
    circle_sigma1 = anomaly_list(1);
    hc1 = anomaly_list(2);
    hc2 = anomaly_list(3);
    hR = anomaly_list(4);
    he1 = anomaly_list(5);
    he2 = anomaly_list(6);

elseif anomaly_type == 2
    circle_sigma1 = anomaly_list(1);
    hc1 = anomaly_list(2);
    hc2 = anomaly_list(3);
    hR1 = anomaly_list(4);
    he1 = anomaly_list(5);
    he2 = anomaly_list(6);
    
    circle_sigma2 = anomaly_list(7);
    hc3 = anomaly_list(8);
    hc4 = anomaly_list(9);
    hR2 = anomaly_list(10);
    he3 = anomaly_list(11);
    he4 = anomaly_list(12);

elseif anomaly_type == 3
    
    circle_sigma1 = anomaly_list(1);
    hc1 = anomaly_list(2);
    hc2 = anomaly_list(3);
    hR1 = anomaly_list(4);
    he1 = anomaly_list(5);
    he2 = anomaly_list(6);
    
    circle_sigma2 = anomaly_list(7);
    hc3 = anomaly_list(8);
    hc4 = anomaly_list(9);
    hR2 = anomaly_list(10);
    he3 = anomaly_list(11);
    he4 = anomaly_list(12);
    
    circle_sigma3 = anomaly_list(13);
    hc5 = anomaly_list(14);
    hc6 = anomaly_list(15);
    hR3 = anomaly_list(16);
    he5 = anomaly_list(17);
    he6 = anomaly_list(18);

else

    circle_sigma1 = anomaly_list(1);
    hc1 = anomaly_list(2);
    hc2 = anomaly_list(3);
    hR1 = anomaly_list(4);
    he1 = anomaly_list(5);
    he2 = anomaly_list(6);
    
    circle_sigma2 = anomaly_list(7);
    hc3 = anomaly_list(8);
    hc4 = anomaly_list(9);
    hR2 = anomaly_list(10);
    he3 = anomaly_list(11);
    he4 = anomaly_list(12);
    
    circle_sigma3 = anomaly_list(13);
    hc5 = anomaly_list(14);
    hc6 = anomaly_list(15);
    hR3 = anomaly_list(16);
    he5 = anomaly_list(17);
    he6 = anomaly_list(18);

    circle_sigma4 = anomaly_list(19);
    hc7 = anomaly_list(20);
    hc8 = anomaly_list(21);
    hR4 = anomaly_list(22);
    he7 = anomaly_list(23);
    he8 = anomaly_list(24);
end

% Conductivities of background is 1
back = 1;
  
% Initialize
[zrow,zcol] = size(z);
z           = z(:);
final_sigma = back*ones(size(z));
x1          = real(z);
x2          = imag(z);

% (hc1,hc2) is the center x,y of the abomaly; 
% (he1, he2) give the eccentrities with respect to radius hR.
% Compute elliptical "distance" of the evaluation points from circle

if anomaly_type == 1

    hd  = sqrt(he1*(x1-hc1).^2 + he2*(x2-hc2).^2);
    final_sigma(hd <= hR) = circle_sigma1;

elseif anomaly_type == 2

    hd  = sqrt(he1*(x1-hc1).^2 + he2*(x2-hc2).^2);
    final_sigma(hd <= hR1) = circle_sigma1;
    
    hd2  = sqrt(he3*(x1-hc3).^2 + he4*(x2-hc4).^2);
    final_sigma(hd2 <= hR2) = circle_sigma2;

elseif anomaly_type == 3
    
    hd  = sqrt(he1*(x1-hc1).^2 + he2*(x2-hc2).^2);
    final_sigma(hd <= hR1) = circle_sigma1;
    
    hd2  = sqrt(he3*(x1-hc3).^2 + he4*(x2-hc4).^2);
    final_sigma(hd2 <= hR2) = circle_sigma2;
    
    hd3  = sqrt(he5*(x1-hc5).^2 + he6*(x2-hc6).^2);
    final_sigma(hd3 <= hR3) = circle_sigma3;

else
    
    hd  = sqrt(he1*(x1-hc1).^2 + he2*(x2-hc2).^2);
    final_sigma(hd <= hR1) = circle_sigma1;
    
    hd2  = sqrt(he3*(x1-hc3).^2 + he4*(x2-hc4).^2);
    final_sigma(hd2 <= hR2) = circle_sigma2;
    
    hd3  = sqrt(he5*(x1-hc5).^2 + he6*(x2-hc6).^2);
    final_sigma(hd3 <= hR3) = circle_sigma3;
    
    hd4  = sqrt(he7*(x1-hc7).^2 + he8*(x2-hc8).^2);
    final_sigma(hd4 <= hR4) = circle_sigma4;

end

% Reshape answer to original form
final_sigma = reshape(final_sigma,[zrow,zcol]);

end
