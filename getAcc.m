%�������ƶȣ���׼ȷ��
function accuracy=getAcc(result1, result2)
    tmp= (result1==result2);
    accuracy=sum(tmp)/length(tmp);
end