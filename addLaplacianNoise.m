
function [noisy_data,lap_noise]=addLaplacianNoise(data,eps, sens)
    [m,n]=size(data);  
    mu=zeros(1,n);
    
    %����Laplacian����
    lap_noise=genLaplasNoise(m, mu, eps, sens ); 
    noisy_data=data+lap_noise;
end