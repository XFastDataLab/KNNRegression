%���ɶ�άlaplacian noise
function [ noise ] = gen_LaplasNoise(m,  MU, eps, sens )
    [row,col]=size(MU);
    
    noise=zeros(m,col);
    
    %ÿ��ά���ϵ�������Laplacian����
    for i=1:col
        cur_noise=gen_one_dim_LaplasNoise(m,  MU(i), eps , sens);
        noise(:,i)=cur_noise;
    end
    
end

%����һάlaplacian noise
function [ noise ] = gen_one_dim_LaplasNoise(m,  mu, eps, sens )
    u=rand(m,1);
    alpha=u-0.5;
    
    sign= (alpha>=0);
    
    lamb=sens/eps;
    
    noise1=  mu -lamb*(sign.*log(1-2*abs(alpha)));
    
    sign= (alpha<0);      
    noise2=  mu+ lamb*(sign.*log(2*abs(alpha)));
    noise=noise1+noise2;
end


