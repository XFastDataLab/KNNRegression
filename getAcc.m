%计算相似度，即准确率
function accuracy=getAcc(result1, result2)
    tmp= (result1==result2);
    accuracy=sum(tmp)/length(tmp);
end