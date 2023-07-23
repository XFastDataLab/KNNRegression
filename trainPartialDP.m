%----------------------------------------------------------------------------
% Author : Yewang Chen
% Email  : ywchen@hqu.edu.cn
% Version:1.0
% Date   :2022/5/27             
% College of Computer Science and Technology @Huaqiao University, Xiamen, China
% Copyright@hqu.edu.cn
%----------------------------------------------------------------------------

% data, labels是总体原始数据及其标签. 
%    1. 将data随机打乱分成不同子集，用于模拟分布式场景，每个子集代表一个节点。
%    2. 对每个数据集分别加拉普拉斯噪声，做knn 回归. 回归后的数据与各自节点上的原数据具有类似的kNN结构。
%    3. 将各个节点的回归数据合并，再放大到与拉普拉斯噪声相同的尺度。 
function [cell_reg_noisy_data,cell_ori_group_data_without_label, cell_new_label,cell_lap_noisy_data_without_label]=trainPartialDP(data,labels,node_num,  eps ,sens,K_train,K_test)
    %惩罚系数
    C=50;
    
    [m,n]=size(data);
    
    %将数据中心化,使得中心为[0 0 ...]
    data_center= sum(data)/m;
    data=data-ones(m,1)*data_center;
    scale=max(max(data))-min(min(data));
    data=sens*data/scale;
    
    figure(1);
    drawshapes(data,labels);
    
    
    %把数据加上标签，以方便在打乱顺序时标签也一起变换
    labeled_data=[data labels];
    
    %将labeled_data随机打乱顺序，平均分成nodes_num个子集，位置 记录在ids中
    [cell_ori_group_data_without_label,cell_group_label]=divide_data_with_label(labeled_data,node_num);
       
    %初始化, 对每个节点进行单独KNN回归，回归结果存储于cell_reg_noisy_data{i}中， 
    %cell_lap_noisydata{i}存储第i个节点的加Laplacian噪声后的数据, 
    cell_lap_noisy_data_without_label={};
    cell_reg_noisy_data={};
    cell_new_label=cell_group_label;
    %分别在不同的节点上做加噪与kNN回归
    
    sum_reg_cov_mat = 0;
    sum_lap_cov_mat = 0;
    sum_reg_avg_simKNN = 0;
    sum_lap_avg_simKNN = 0;
    cell_lap_noise = {};
    cell_reg_noise = {};
    for i=1:node_num
        %取到当前子集数据
        %cur_data=divided_data(divided_ids(i)+1:divided_ids(i+1),:);
        cur_data=cell_ori_group_data_without_label{i};
        cur_label=cell_group_label{i};
        
        [noisy_data,lap_noise,reg_noise,reg_avg_simKNN,lap_avg_simKNN]= regKNNNoise (cur_data,C,K_train,eps,sens) ;
        cell_lap_noise{i} = lap_noise;
        cur_lap_noisydata=cur_data+lap_noise;
        cell_lap_noisy_data_without_label{i}=cur_lap_noisydata;
        cur_lap_scale= max(max(cur_lap_noisydata))-min(min(cur_lap_noisydata));
        cur_reg_scale= max(max(noisy_data))-min(min(noisy_data));
        noisy_data=noisy_data*(cur_lap_scale/cur_reg_scale);
        
        enlarge_reg_noise = noisy_data - cur_data;
        cell_reg_noise{i} = enlarge_reg_noise;
        
        cell_reg_noisy_data{i}=noisy_data;
        figure(3+i)
        drawshapes(noisy_data,cur_label);
        figure(1+i)
        drawshapes(cur_lap_noisydata,cur_label);
        figure(5+i)
        drawshapes(cur_data,cur_label);
        
        %获取协方差矩阵
        reg_cov_mat=get_cov_mat(noisy_data);
        lap_cov_mat=get_cov_mat(cur_lap_noisydata);
        ori_cov_mat = get_cov_mat(cur_data);
        ori_cov_mat = det(ori_cov_mat);
        reg_cov_mat = det(reg_cov_mat);
        lap_cov_mat = det(lap_cov_mat);
        reg_cov_mat = reg_cov_mat/ori_cov_mat;
        lap_cov_mat = lap_cov_mat/ori_cov_mat;
        sum_reg_cov_mat = sum_reg_cov_mat + reg_cov_mat;
        sum_lap_cov_mat = sum_lap_cov_mat + lap_cov_mat;
        
        %获取总的simKNN
        sum_reg_avg_simKNN = sum_reg_avg_simKNN + reg_avg_simKNN;
        sum_lap_avg_simKNN = sum_lap_avg_simKNN + lap_avg_simKNN;
         
    end
    for i=1:length(cell_lap_noise)
        a = cell_lap_noise(i);
        b = cell_reg_noise(i);
        a = cell2mat(a);
        b = cell2mat(b);
        %save t48k_lap_noise.txt a -ascii -append;
        %save t48k_reg_noise.txt b -ascii -append;
    end

    
    %平均协方差噪声
    mean_reg_cov_mat = sum_reg_cov_mat/node_num;
    mean_lap_cov_mat = sum_lap_cov_mat/node_num;
    mean_reg_cov_mat
    mean_lap_cov_mat
    
    %平均simKNN
    mean_reg_avg_simKNN = sum_reg_avg_simKNN/node_num;
    mean_lap_avg_simKNN = sum_lap_avg_simKNN/node_num;
    mean_reg_avg_simKNN
    mean_lap_avg_simKNN

end



%获取数据的协方差矩阵 
function [cov_mat]=get_cov_mat(data)
    [m,n]=size(data);
    mu=sum(data)/m;
    cov_mat=zeros(n,n);
    for i=1:m
        cur_p=data(i,:)-mu;
        cov_mat=cov_mat+ cur_p'*cur_p;
    end
    cov_mat=cov_mat/m;
end

%reg_noisy_data:回归后的加噪数据
%old_lap_noise :原始噪声
%reg_noise     :回归噪声
function [reg_noisy_data,old_lap_noise,reg_noise,reg_avg_simKNN,lap_avg_simKNN]=regKNNNoise (data,C,K,eps, sens) 
    [m,n]=size(data);
    delta=max(max(data))-min(min(data));
    
    mu=zeros(1,n);
    
    %生成Laplacian噪声
    old_lap_noise=genLaplasNoise(m, mu, eps, sens );    
   
    reg_noise=old_lap_noise;
    
    ori_kNeibMatrix=zeros(m,m);
    
    %得到原始数据的近邻矩阵 
    ori_kNeibMatrix=update_kNeibMatrix(ori_kNeibMatrix,data,K);
        
    noisy_kNeibMatrix=zeros(m,m);
    reg_noisy_data=data+reg_noise;
    
    %得到加噪数据的近邻矩阵 
    noisy_kNeibMatrix=update_kNeibMatrix(noisy_kNeibMatrix,reg_noisy_data,K);
    
    lap_kNeibMatrix=noisy_kNeibMatrix;
    %迭代次数
    iter=200;
    
    %每次迭代步长
    alpha=0.01;
        

    for it=1: iter
        for i=1:m
            %计算梯度
            cur_grad=getGrad(i,reg_noisy_data, reg_noise,ori_kNeibMatrix,noisy_kNeibMatrix,C);
            %更新噪声
            reg_noise(i,:)=reg_noise(i,:)-alpha*cur_grad;
            %更新加噪数据
            reg_noisy_data(i,:)=data(i,:)+reg_noise(i,:);
        end
        
        %重新更新近邻矩阵 
        noisy_kNeibMatrix=update_kNeibMatrix(noisy_kNeibMatrix,reg_noisy_data,K);
    end
    
    [reg_avg_simKNN]=average_sim(ori_kNeibMatrix, noisy_kNeibMatrix,K);
    
    [lap_avg_simKNN]=average_sim(ori_kNeibMatrix, lap_kNeibMatrix,K);
    
end 

%随机打乱data顺序
function [ new_data ] = disorder( data )
    [m,n]=size(data);
    ids = randperm(m);                  %生成1-m个不重复的整数
    new_data=zeros(m,n);
    for i=1:m
        new_data(ids(i),:)=data(i,:);   %换位置 
    end
end

%传入带标签的数据，打乱顺序分成num份，每份代表一个节点，
%cell_group_data_without_label{i}表示第i个节点数据（不带标签），
%cell_group_label{i}表示第i个节点数据的标签
function [cell_group_data_without_label,cell_group_label]=divide_data_with_label(data, num)
    [cell_group_data]=divide_data(data, num);
    for i=1:num
        cur_cell_data=cell_group_data{i};
        [m,n]=size(cur_cell_data);
        cell_group_data_without_label{i}=cur_cell_data(:,1:n-1);
        cell_group_label{i}=cur_cell_data(:,n);
    end
end

%将数据分为若干个子集, ids标记出每个子集的位置，如
%ids=[0 5 10 16] 表示，第一个数据集为new_data中从0+1到5, 第2个数据集为new_data从5+1到10,
%第三个数据集从10+1到16.
function [cell_group_data,new_order_data]=divide_data(data, num)
    %随机打乱数据顺序
    new_order_data = disorder(data);
    cell_group_data={};
    %m is row, n is col
    [m,n]=size(data); 
    
    each_num=ceil(m/num);

    loc=1;
    for i=1:num-1;
        cell_group_data{i}=new_order_data(loc:loc+each_num-1,:);
        loc=loc+each_num;
    end
    cell_group_data{num}=new_order_data(loc:m,:);   
end


%对第i个点计算J(noise_i)计算梯度
function grad=getGrad(i,noisy_data, noise,ori_kNeibMatrix,noisy_kNeibMatrix,C)
    [m,n]=size(noisy_kNeibMatrix);
    cur_noise=noise(i,:);
    grad=-C*exp(-norm(cur_noise)^2)*cur_noise;
     
    for j=1:m
        %如果之前j是i的近邻，现在不是
        if (ori_kNeibMatrix(i,j)-noisy_kNeibMatrix(i,j))==1
            grad=grad+ noisy_data(i,:)-noisy_data(j,:);         
        end
        
        %如果之前i是j的近邻，现在不是
        if (ori_kNeibMatrix(j,i)-noisy_kNeibMatrix(j,i))==1
            grad=grad+ noisy_data(i,:)-noisy_data(j,:);             
        end
    end    
end

%计算两个重向量相似度
function [same_ele_number]=same_elements(knn_neibs1, knn_neibs2)
    compare_v=(knn_neibs1.*knn_neibs2);    
    same_ele_number=sum(compare_v);
end

%计算KNN结构相似度
function [avg_sim]=average_sim(ori_kNeibMatrix, tmp_kNeibMatrix,K)
    [m,n]=size(ori_kNeibMatrix);

    same_elements_array=zeros(1,m);
    for i=1:m
        knn_neibs1=ori_kNeibMatrix(i,:);
        knn_neibs2=tmp_kNeibMatrix(i,:); 
        same_elements_array(i)=same_elements(knn_neibs1, knn_neibs2);        
    end
    avg_sim=(sum(same_elements_array)-m)/(m*(K-1));
end

%更新k近邻标识矩阵 
function kNeibMatrix=update_kNeibMatrix(kNeibMatrix,data,K)    
    [m,n]=size(data);
    for i=1:m
        dists=pdist2(data(i,:),data);
        [sort_dists,sort_ids]=sort(dists);
        kNeibMatrix(i,:)=zeros(1,m);
        kNeibMatrix(i,sort_ids(1:K))=1;
    end
end 

%更新矩阵R 
function mat_R=update_Mat_R(new_data,ori_data)    
    [m,n]=size(ori_data);
    mat_R=zeros(m,m);
    for i=1:m
        ori_dists=pdist2(ori_data(i,:),ori_data);
        new_dists=pdist2(new_data(i,:),new_data);
        mat_R(i,:)=ori_dists./new_dists;
        mat_R(i,i)=1;
    end
end 


