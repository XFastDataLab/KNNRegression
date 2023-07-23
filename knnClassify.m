
%按kNN对test_data进行投票分类，再于给定答案标签进行比较.
function [ accuracy,test_labels] = knnClassify( train_data,trained_labels, test_data, test_labels, K )
    [m,n]=size(test_data);
    
    %分类结果标签
    result_labels=-1*ones(m,1);
    dists=pdist2(train_data,test_data);
    
    unique_label=unique(trained_labels);
    label_num=length(unique_label);
    
    for i=1:m
        %第i列是test_data中第i个点到train_data中的所有点的距离
        cur_dists=dists(:,i);       
        [sorted_cur_dist,sorted_id]=sort(cur_dists);
        first_k_ids=sorted_id(1:K);
        knn_labels=trained_labels(first_k_ids);
        
        label_count_num_v=zeros(label_num,1);
        %统计label 个数
        for j=1:label_num
            cur_lab=unique_label(j);
            %等于当前label的个数
            count_num=length(find(knn_labels==cur_lab));
            label_count_num_v(j)=count_num;
        end
        
        [max_v, max_label_id]=max(label_count_num_v);
        
        %分配类别
        result_labels(i)=unique_label(max_label_id);
    end
    
    accuracy= getAcc(test_labels, result_labels);
end

%计算相似度，即准确率
function accuracy=getAcc(result1, result2)
    tmp= (result1==result2);
    accuracy=sum(tmp)/length(tmp);
end
