
%��kNN��test_data����ͶƱ���࣬���ڸ����𰸱�ǩ���бȽ�.
function [ accuracy,test_labels] = knnClassify( train_data,trained_labels, test_data, test_labels, K )
    [m,n]=size(test_data);
    
    %��������ǩ
    result_labels=-1*ones(m,1);
    dists=pdist2(train_data,test_data);
    
    unique_label=unique(trained_labels);
    label_num=length(unique_label);
    
    for i=1:m
        %��i����test_data�е�i���㵽train_data�е����е�ľ���
        cur_dists=dists(:,i);       
        [sorted_cur_dist,sorted_id]=sort(cur_dists);
        first_k_ids=sorted_id(1:K);
        knn_labels=trained_labels(first_k_ids);
        
        label_count_num_v=zeros(label_num,1);
        %ͳ��label ����
        for j=1:label_num
            cur_lab=unique_label(j);
            %���ڵ�ǰlabel�ĸ���
            count_num=length(find(knn_labels==cur_lab));
            label_count_num_v(j)=count_num;
        end
        
        [max_v, max_label_id]=max(label_count_num_v);
        
        %�������
        result_labels(i)=unique_label(max_label_id);
    end
    
    accuracy= getAcc(test_labels, result_labels);
end

%�������ƶȣ���׼ȷ��
function accuracy=getAcc(result1, result2)
    tmp= (result1==result2);
    accuracy=sum(tmp)/length(tmp);
end
