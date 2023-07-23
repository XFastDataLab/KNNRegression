function error_label = get_error_label( result1, result2 )
  index = find(result1~=result2);
  error_label = [index result2(index)];
  acc_label = [index result1(index)];
  all_label = [index result1(index), result2(index)];
  %save NoN1_error_label.txt all_label -ascii -append;
  %save NoN1_acc_label.txt acc_label -ascii -append; 
end

