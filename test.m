function test()

% Toy Data
     train_data=load('dataset/Toy_Train_Data.txt');
     test_data=load('dataset/Toy_Test_Data.txt');
     train_label=load('dataset/Toy_Ytrain_Data.txt');
     test_label=load('dataset/Toy_Ytest_Data.txt');
     eps =1;
     sens=20;
     K_train=15;
     K_test=10;
     node_num=2;
    
% Aggregation
%     train_data=load('dataset/Agg_Train_Data.txt');
%     test_data=load('dataset/Agg_Test_Data.txt');
%     train_label=load('dataset/Agg_Ytrain_Data.txt');
%     test_label=load('dataset/Agg_Ytest_Data.txt');
% 
%     eps =1;
%     sens=20;
%     K_train=15;
%     K_test=10;
%     node_num=10;
    
% Spiral
%     train_data=load('dataset/Spiral_Train_Data.txt');
%     test_data=load('dataset/Spiral_Test_Data.txt');
%     train_label=load('dataset/Spiral_Ytrain_Data.txt');
%     test_label=load('dataset/Spiral_Ytest_Data.txt');
% 
%     eps =1;
%     sens=20;
%     K_train=10;
%     K_test=20;
%     node_num=3;

% Phoneme
%     train_data=load('dataset/Phoneme_Train_Data.txt');
%     test_data=load('dataset/Phoneme_Test_Data.txt');
%     train_label=load('dataset/Phoneme_Ytrain_Data.txt');
%     train_label=train_label;
%     test_label=load('dataset/Phoneme_Ytest_Data.txt');
%     test_label=test_label;
%     eps =1;
%     sens=20;
%     K_train=10;
%     K_test=10;
%     node_num=10;

%banknote
%     train_data=load('dataset/Banknote_Train_Data.txt');
%     test_data=load('dataset/Banknote_Test_Data.txt');
%     train_label=load('dataset/Banknote_Ytrain_Data.txt');
%     train_label=train_label;
%     test_label=load('dataset/Banknote_Ytest_Data.txt');
%     test_label=test_label;
%     eps =1;
%     sens=20;
%     K_train=5;
%     K_test=10;
%     node_num=5;

% Balance
%     train_data=load('dataset/Balance_Train_Data.txt');
%     test_data=load('dataset/Balance_Test_Data.txt');
%     train_label=load('dataset/Balance_Ytrain_Data.txt');
%     test_label=load('dataset/Balance_Ytest_Data.txt');
%     eps =1;
%     sens=20;
%     K_train=5;
%     K_test=10;
%     node_num=4;

%Banana
%     train_data=load('dataset/Banana_Train_Data.txt');
%     test_data=load('dataset/Banana_Test_Data.txt');
%     train_label=load('dataset/Banana_Ytrain_Data.txt');
%     train_label=train_label;
%     test_label=load('dataset/Banana_Ytest_Data.txt');
%     test_label=test_label;
%     eps = 1;
%     sens= 20;
%     K_train = 5;
%     K_test = 10;
%     node_num=10;

% t48k
   % train_data=load('dataset/t48k_Train_Data.txt');
   % test_data=load('dataset/t48k_Test_Data.txt');
   % train_label=load('dataset/t48k_Ytrain_Data.txt');
   % test_label=load('dataset/t48k_Ytest_Data.txt');
   % eps =1;
   % sens=20;
   % K_train=15;
   % K_test=10;
   % node_num=10;
    
    %cell_reg_noisy_data{i}�洢��i���ڵ��ϵļ���������
    %cell_group_data_without_label{i}�洢��i���ڵ��ϵ�����
    %cell_new_label{i}�ǵ�i���ڵ�����ݱ�ǩ.
    %cell_lap_noisy_data_without_label{i}�洢��i���ڵ��ϵļ�Laplacian�������ԭʼ����
    
    [cell_reg_noisy_data, cell_group_data_without_label,cell_new_label,cell_lap_noisy_data_without_label]=trainPartialDP(train_data,train_label,node_num, eps ,sens,K_train, K_test);
    
    %��ÿ���ڵ��ԭʼ���� ���ֱ����ڵ��еĸ�������� 
    [cell_ori_class_center,cell_class_label]=get_centers_for_all_nodes(cell_group_data_without_label,cell_new_label,node_num);
    
    %�����������ݱ��Ŵ������ɱ�����ԭʼ���ݷŴ���ͬ�߶Ⱥ��ܽϲŽ���ƽ�ƱȽϡ������ں�����λ�Ʋ����л����������
    times_to_enlarge=sens/(2*eps);
    cell_ori_class_center_enlarged=enlarge(cell_ori_class_center,times_to_enlarge);
    
    %��ÿ���ڵ�ļ����������� ���ֱ����ڵ��еĸ��������
    [cell_reg_noisy_class_center,cell_class_label]=get_centers_for_all_nodes(cell_reg_noisy_data,cell_new_label,node_num);
     for i=1:node_num
         figure(i+3)
         cur_cell_centers=cell_reg_noisy_class_center{i};
         m_c=length(cur_cell_centers);
         for j=1:m_c
             tmp_center=cur_cell_centers{j};            
             plot(tmp_center(:,1),tmp_center(:,2),'ro');
             set(gcf,'unit','normalized','position',[.4 .4 .2 .30])
             set(gca,'FontSize',15);
         end
     end
    

    [m_test,n]=size(test_data);
    
    mu=zeros(1,n);
    %test_data=test_data*times_to_enlarge;
    result_labels=zeros(m_test,1);
    
    labels_of_each_node=zeros(m_test,node_num);
    for i=1:m_test
        %�����һ���ڵ�,������ڵ�������i��test
        cur_node=ceil(rand(1,1)*node_num);
        cur_test_data=test_data(i,:);
        
        cur_cell_ori_node_data=cell_group_data_without_label{cur_node};
        cur_label= cell_new_label{cur_node};
%        figure(6);
%        drawshapes(cur_cell_ori_node_data,cur_label);   
%         hold on
%         plot(cur_test_data(:,1),cur_test_data(:,2),'k*')
        
        cur_cell_ori_node_data=cell_group_data_without_label{cur_node}*times_to_enlarge;
        cur_cell_ori_class_center=cell_ori_class_center_enlarged{cur_node};                           
       
        cur_test_data=cur_test_data*times_to_enlarge;
        
%         figure(7);
%         drawshapes(cur_cell_ori_node_data,cur_label);   
%         hold on
%         plot(cur_test_data(:,1),cur_test_data(:,2),'k*')
%         set(gcf,'unit','normalized','position',[.4 .4 .2 .30])
         
        %plot (cur_cell_ori_class_center(:,1),cur_cell_ori_class_center(:,2),'ro')
%         draw_cell_centers(cur_cell_ori_class_center);
        
        %cur_cell_reg_noisy_class_center=cell_reg_noisy_class_center{cur_node};
        %[m_c,n]=size(cur_cell_ori_class_center);
        cur_cell_class_label=cell_class_label{cur_node};
        m_c=length(cur_cell_class_label);
                
        %֧�־���suport_mat(j,t)=s ��ʾ��j���ڵ�����֧��cur_test_data ���ڵ�t�����Ľ��ڵ����
        suport_mat=zeros(node_num,m_c);
        for j=1:node_num            
            %figure(3+j);            
            %ѡ�������ڵ�
            tmp_cell_reg_noisy_data=cell_reg_noisy_data{j};
            %ȡ����ǰ��ʱ���������ݸ��������
            tmp_cell_reg_noisy_class_center=cell_reg_noisy_class_center{j};
              
            tmp_cell_new_label=cell_new_label{j};
            
%             figure(j+3)
%             drawshapes(tmp_cell_reg_noisy_data,tmp_cell_new_label);   
%             hold on 
            %plot (tmp_cell_reg_noisy_class_center(:,1),tmp_cell_reg_noisy_class_center(:,2),'ro')
            %draw_cell_centers(tmp_cell_reg_noisy_class_center);
            
            %���㵱ǰ�ڵ�ĸ����������tmp_cell_reg_noisy_class_center��,�� tmp_shift_v(t)Ϊ��j���ڵ�ĵ�t�����noisy data����-��i���ڵ�ĵ�t�����ԭʼ��������
            %tmp_shift_v=tmp_cell_reg_noisy_class_center-cur_cell_ori_class_center;
            tmp_shift_v=minus_cell(tmp_cell_reg_noisy_class_center,cur_cell_ori_class_center);
            
            %��ÿ�������в��ԣ� ����cur_test_data�����ڵ�t�������ô��������λ�ƣ�tmp_shift_v(t)
            %���ڵ�ǰ�ڵ��noisy data�н����жϣ��ҵ��ж��ٸ�����֧�������ڸ��ࡣ
            for t=1:m_c
                %tmp_fake_test_data=cur_test_data+tmp_shift_v(t,:);
                tmp_fake_test_data=cur_test_data+tmp_shift_v{t};
                %plot(tmp_fake_test_data(:,1),tmp_fake_test_data(:,2),'k^')
                kNeib_local_ids=findKNN(tmp_cell_reg_noisy_data,tmp_fake_test_data,K_test);
                tmp_labels=tmp_cell_new_label(kNeib_local_ids);
                %���ٸ�֧�ֵ�
                num_of_suporter=length(find(tmp_labels==cur_cell_class_label(t)));
                suport_mat(j,t)=num_of_suporter;
            end
            
            [max_v, max_label_id]=max(suport_mat(j,:));
            labels_of_each_node(i,j)=max_label_id;
        end
        
        tmp_suport_v=suport_mat;
        if node_num>1
            tmp_suport_v=sum(tmp_suport_v);
        end             
        
        
        %��¼�ڵ�ǰ�ڵ��ϵķ���
        [max_v, max_label_id]=max(tmp_suport_v);
        result_labels(i)=cur_cell_class_label(max_label_id);
    end
    
    %��ÿ������ϣ��ֱ𵥶��������в��Ե��׼ȷ��
    for j=1:node_num
        node_accuracy(j)= getAcc(test_label,labels_of_each_node(:,j))
    end
    total = 0;
    all_label = [test_label labels_of_each_node];
    n = length(test_label);
    m = node_num+1;
    for i = 1:n
        tag = 0;
        for j = 1:m-1
            if all_label(i,j) == all_label(i,j+1)
              tag = tag + 1;
            end
        end
        if tag == 6
            total = total + 1;
        end
    end
    save all_label.txt all_label -ascii -append;
    %����result_labels������׼ȷ��
    overall_mean_accuracy= getAcc(test_label,result_labels)
    total
    
    %��ÿ���ڵ��ϲ��Դ���Laplacian�������
    lap_noisy_test_data=addLaplacianNoise(test_data,eps, sens);
    lap_accuracy=zeros(node_num,1);
    for j=1:node_num       
        local_lap_noisy_trained_data=cell_lap_noisy_data_without_label{j};
        tmp_cell_new_label=cell_new_label{j};
        lap_accuracy(j)= knnClassify( local_lap_noisy_trained_data,tmp_cell_new_label, lap_noisy_test_data, test_label, K_test) ;   
    end
        
    mean_lap_accuracy=mean(lap_accuracy)
end

%��ÿ���ڵ����ݼ���ÿ���������ĵ�
function   [cell_reg_noisy_class_center,cell_class_label]= get_centers_for_all_nodes(cell_reg_noisy_data,cell_new_label,node_num)
    cell_reg_noisy_class_center={};
    cell_class_label={};
    for i=1:node_num        
        cur_data=cell_reg_noisy_data{i};
        cur_labels=cell_new_label{i};
        uni_labels=unique(cur_labels);
        uni_labels=sort(uni_labels);
        cur_centers={};
        min_label=min(uni_labels);
        max_label=max(uni_labels);
        
        %����������ж�
        cur_loc=1;
        for j=min_label:max_label
            ids=find(cur_labels==j);
            tmp_center=mean(cur_data(ids,:));
            cur_centers{cur_loc}=tmp_center;
            cur_loc=cur_loc+1;
        end
        cell_reg_noisy_class_center{i}=cur_centers;
        cell_class_label{i}=min_label:max(uni_labels);
    end
end

function [cell_enlarged]=enlarge(acell,times_to_enlarge)
    m=length(acell);
    cell_enlarged={};
    for i=1:m
        tmp_cell=acell{i};
        n=length(tmp_cell);
        for j=1:n
           tmp_cell{j}=tmp_cell{j}*times_to_enlarge;
        end        
        cell_enlarged{i}=tmp_cell;
    end
end

function draw_cell_centers(cell_class_center)
    m=length(cell_class_center);
    for i=1:m
        tmp_cell=cell_class_center{i};       
        plot(tmp_cell(:,1),tmp_cell(:,2),'ro')          
    end
end

function [result]=minus_cell(cell1, cell2)
    len1=length(cell1);
    result={};
    for i=1:len1
        result{i}=cell1{i}-cell2{i};
    end
end