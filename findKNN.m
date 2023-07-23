%²éÕÒK½üÁÚ
function [k_neibs_mat,dists_mat]=findkNN(source_data, query_data, K)
    dist_mat=pdist2(source_data,query_data);
    [m,n]=size(query_data);
    dists_mat=zeros(m,K);
    k_neibs_mat=zeros(m,K);
    for i=1:m
        cur_dists=dist_mat(:,i);
        [sorted_dists, sorted_ids]=sort(cur_dists);
        dists_mat(i,:)=sorted_dists(1:K,i)';
        k_neibs_mat(i,:)=sorted_ids(1:K,i)';
    end
end