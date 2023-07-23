%----------------------------------------------------------------------------
% Author : Yewang Chen
% Email  : ywchen@hqu.edu.cn
% Version:1.0
% Date   :2022/5/27             
% College of Computer Science and Technology @Huaqiao University, Xiamen, China
% Copyright@hqu.edu.cn
%----------------------------------------------------------------------------

% data, labels������ԭʼ���ݼ����ǩ. 
%    1. ��data������ҷֳɲ�ͬ�Ӽ�������ģ��ֲ�ʽ������ÿ���Ӽ�����һ���ڵ㡣
%    2. ��ÿ�����ݼ��ֱ��������˹��������knn �ع�. �ع�����������Խڵ��ϵ�ԭ���ݾ������Ƶ�kNN�ṹ��
%    3. �������ڵ�Ļع����ݺϲ����ٷŴ���������˹������ͬ�ĳ߶ȡ� 
function [cell_reg_noisy_data,cell_ori_group_data_without_label, cell_new_label,cell_lap_noisy_data_without_label]=trainPartialDP(data,labels,node_num,  eps ,sens,K_train,K_test)
    %�ͷ�ϵ��
    C=50;
    
    [m,n]=size(data);
    
    %���������Ļ�,ʹ������Ϊ[0 0 ...]
    data_center= sum(data)/m;
    data=data-ones(m,1)*data_center;
    scale=max(max(data))-min(min(data));
    data=sens*data/scale;
    
    figure(1);
    drawshapes(data,labels);
    
    
    %�����ݼ��ϱ�ǩ���Է����ڴ���˳��ʱ��ǩҲһ��任
    labeled_data=[data labels];
    
    %��labeled_data�������˳��ƽ���ֳ�nodes_num���Ӽ���λ�� ��¼��ids��
    [cell_ori_group_data_without_label,cell_group_label]=divide_data_with_label(labeled_data,node_num);
       
    %��ʼ��, ��ÿ���ڵ���е���KNN�ع飬�ع����洢��cell_reg_noisy_data{i}�У� 
    %cell_lap_noisydata{i}�洢��i���ڵ�ļ�Laplacian�����������, 
    cell_lap_noisy_data_without_label={};
    cell_reg_noisy_data={};
    cell_new_label=cell_group_label;
    %�ֱ��ڲ�ͬ�Ľڵ�����������kNN�ع�
    
    sum_reg_cov_mat = 0;
    sum_lap_cov_mat = 0;
    sum_reg_avg_simKNN = 0;
    sum_lap_avg_simKNN = 0;
    cell_lap_noise = {};
    cell_reg_noise = {};
    for i=1:node_num
        %ȡ����ǰ�Ӽ�����
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
        
        %��ȡЭ�������
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
        
        %��ȡ�ܵ�simKNN
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

    
    %ƽ��Э��������
    mean_reg_cov_mat = sum_reg_cov_mat/node_num;
    mean_lap_cov_mat = sum_lap_cov_mat/node_num;
    mean_reg_cov_mat
    mean_lap_cov_mat
    
    %ƽ��simKNN
    mean_reg_avg_simKNN = sum_reg_avg_simKNN/node_num;
    mean_lap_avg_simKNN = sum_lap_avg_simKNN/node_num;
    mean_reg_avg_simKNN
    mean_lap_avg_simKNN

end



%��ȡ���ݵ�Э������� 
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

%reg_noisy_data:�ع��ļ�������
%old_lap_noise :ԭʼ����
%reg_noise     :�ع�����
function [reg_noisy_data,old_lap_noise,reg_noise,reg_avg_simKNN,lap_avg_simKNN]=regKNNNoise (data,C,K,eps, sens) 
    [m,n]=size(data);
    delta=max(max(data))-min(min(data));
    
    mu=zeros(1,n);
    
    %����Laplacian����
    old_lap_noise=genLaplasNoise(m, mu, eps, sens );    
   
    reg_noise=old_lap_noise;
    
    ori_kNeibMatrix=zeros(m,m);
    
    %�õ�ԭʼ���ݵĽ��ھ��� 
    ori_kNeibMatrix=update_kNeibMatrix(ori_kNeibMatrix,data,K);
        
    noisy_kNeibMatrix=zeros(m,m);
    reg_noisy_data=data+reg_noise;
    
    %�õ��������ݵĽ��ھ��� 
    noisy_kNeibMatrix=update_kNeibMatrix(noisy_kNeibMatrix,reg_noisy_data,K);
    
    lap_kNeibMatrix=noisy_kNeibMatrix;
    %��������
    iter=200;
    
    %ÿ�ε�������
    alpha=0.01;
        

    for it=1: iter
        for i=1:m
            %�����ݶ�
            cur_grad=getGrad(i,reg_noisy_data, reg_noise,ori_kNeibMatrix,noisy_kNeibMatrix,C);
            %��������
            reg_noise(i,:)=reg_noise(i,:)-alpha*cur_grad;
            %���¼�������
            reg_noisy_data(i,:)=data(i,:)+reg_noise(i,:);
        end
        
        %���¸��½��ھ��� 
        noisy_kNeibMatrix=update_kNeibMatrix(noisy_kNeibMatrix,reg_noisy_data,K);
    end
    
    [reg_avg_simKNN]=average_sim(ori_kNeibMatrix, noisy_kNeibMatrix,K);
    
    [lap_avg_simKNN]=average_sim(ori_kNeibMatrix, lap_kNeibMatrix,K);
    
end 

%�������data˳��
function [ new_data ] = disorder( data )
    [m,n]=size(data);
    ids = randperm(m);                  %����1-m�����ظ�������
    new_data=zeros(m,n);
    for i=1:m
        new_data(ids(i),:)=data(i,:);   %��λ�� 
    end
end

%�������ǩ�����ݣ�����˳��ֳ�num�ݣ�ÿ�ݴ���һ���ڵ㣬
%cell_group_data_without_label{i}��ʾ��i���ڵ����ݣ�������ǩ����
%cell_group_label{i}��ʾ��i���ڵ����ݵı�ǩ
function [cell_group_data_without_label,cell_group_label]=divide_data_with_label(data, num)
    [cell_group_data]=divide_data(data, num);
    for i=1:num
        cur_cell_data=cell_group_data{i};
        [m,n]=size(cur_cell_data);
        cell_group_data_without_label{i}=cur_cell_data(:,1:n-1);
        cell_group_label{i}=cur_cell_data(:,n);
    end
end

%�����ݷ�Ϊ���ɸ��Ӽ�, ids��ǳ�ÿ���Ӽ���λ�ã���
%ids=[0 5 10 16] ��ʾ����һ�����ݼ�Ϊnew_data�д�0+1��5, ��2�����ݼ�Ϊnew_data��5+1��10,
%���������ݼ���10+1��16.
function [cell_group_data,new_order_data]=divide_data(data, num)
    %�����������˳��
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


%�Ե�i�������J(noise_i)�����ݶ�
function grad=getGrad(i,noisy_data, noise,ori_kNeibMatrix,noisy_kNeibMatrix,C)
    [m,n]=size(noisy_kNeibMatrix);
    cur_noise=noise(i,:);
    grad=-C*exp(-norm(cur_noise)^2)*cur_noise;
     
    for j=1:m
        %���֮ǰj��i�Ľ��ڣ����ڲ���
        if (ori_kNeibMatrix(i,j)-noisy_kNeibMatrix(i,j))==1
            grad=grad+ noisy_data(i,:)-noisy_data(j,:);         
        end
        
        %���֮ǰi��j�Ľ��ڣ����ڲ���
        if (ori_kNeibMatrix(j,i)-noisy_kNeibMatrix(j,i))==1
            grad=grad+ noisy_data(i,:)-noisy_data(j,:);             
        end
    end    
end

%�����������������ƶ�
function [same_ele_number]=same_elements(knn_neibs1, knn_neibs2)
    compare_v=(knn_neibs1.*knn_neibs2);    
    same_ele_number=sum(compare_v);
end

%����KNN�ṹ���ƶ�
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

%����k���ڱ�ʶ���� 
function kNeibMatrix=update_kNeibMatrix(kNeibMatrix,data,K)    
    [m,n]=size(data);
    for i=1:m
        dists=pdist2(data(i,:),data);
        [sort_dists,sort_ids]=sort(dists);
        kNeibMatrix(i,:)=zeros(1,m);
        kNeibMatrix(i,sort_ids(1:K))=1;
    end
end 

%���¾���R 
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


