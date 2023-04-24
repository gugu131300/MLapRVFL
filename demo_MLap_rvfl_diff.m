%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%MLap%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
seed = 12345678;

load('F:\master1\gxy_paper1\matlab代码\2878\labels.mat');
load('F:\master1\gxy_paper1\matlab代码\2878\X.mat');
train_X = [X];
y = [labels];
%y(find(y==2)) = -1;
xx = line_map(train_X);

Lap_parameter.MaxIterations = 5;
Lap_parameter.gamma = 2^-1;
Lap_parameter.k_nn = 5;
Lap_parameter.ro = 1.5;
Lap_parameter.isMv = 1;

option.N = 100;
option.n_center_vec = 140;
option.seed = 12345678;
option.lambda_1 = 0.1;
option.lambda_2 = 1;
option.sigma = 0.1;
option.type_task = 'binary classification';
option.lammda = 0.01

nfolds = 5;
predict_y_all = y;
predict_y_all = 0;
crossval_idx = crossvalind('Kfold',y(:),nfolds);
RMSE_v = [];R_2=[];Adjusted_R=[];dim = size(xx,2);
ACC=[];SN=[];Spec=[];PE=[];NPV=[];F_score=[];MCC=[];AUC=[];
        
 for fold=1:nfolds

    train_idx = find(crossval_idx~=fold);
    test_idx = find(crossval_idx==fold);
    train_y = y(train_idx,1);
    train_x = xx(train_idx,:); 

    test_y = y(test_idx,1);
    test_x = xx(test_idx,:);
    [results] = MvLap_rvfl2(train_x,train_y,test_x,test_y,option,Lap_parameter);
   
    predict_y = results.Y;
    values = results.values;
    [ACC_i,SN_i,Spec_i,PE_i,NPV_i,F_score_i,MCC_i] = roc2(predict_y,test_y);
    score_s = 1./(1.0+exp(-1.0 * values));
    %[mcc_list,acc_lst,sn_list,sp_list] = Threhold_Find(score_s,test_y,0.1,0.9,0.0001);
    [X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,score_s(:,1),2);
    ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];AUC=[AUC,AUC_i];
    
    break;

 end


MLapRVFL = plot(X_point,Y_point,'Color',[4 157 107]/255,'linewidth',2);
hold on;
frame = getframe(gcf);
img=frame2im(frame);
title('AOUROC');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SVM%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('F:\master1\gxy_paper1\matlab代码\2878\labels.mat');
load('F:\master1\gxy_paper1\matlab代码\2878\X.mat');
train_X = [X];
y = [labels];
xx = line_map(train_X);
nfolds = 3;
predict_y_all = y;
predict_y_all = 0;
crossval_idx = crossvalind('Kfold',y(:),nfolds);
RMSE_v = [];R_2=[];Adjusted_R=[];dim = size(xx,2);
ACC=[];SN=[];Spec=[];PE=[];NPV=[];F_score=[];MCC=[];

 for fold=1:nfolds
     
    %%%划分训练集和测试集
    train_idx = find(crossval_idx~=fold);
    test_idx = find(crossval_idx==fold);
    train_y = y(train_idx,1);
    train_x = xx(train_idx,:); 
    test_y = y(test_idx,1);
    test_x = xx(test_idx,:);
     
    model_ori = fitcsvm(train_x, train_y,'Holdout',0.1);
    model_new = model_ori.Trained{1};

    %testInd = test(model_ori.Partition);
    %dataTest = train_X(testInd,:);
    %labelTest = train_label(testInd,:);
    [labelpredict,scores] = predict(model_new, test_x);
    ACC=[];SN=[];Spec=[];PE=[];NPV=[];F_score=[];MCC=[];AUC=[];
    [ACC_i,SN_i,Spec_i,PE_i,NPV_i,F_score_i,MCC_i] = roc2(labelpredict,test_y );
    [X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,scores(:,2),2);
    AUC=[AUC,AUC_i];ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];


 end

SVM = plot(X_point,Y_point,'Color',[65 105 225]/255,'linewidth',2);
frame = getframe(gcf);
img=frame2im(frame);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%RF%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%  导入数据
load('F:\master1\gxy_paper1\matlab代码\2878\labels.mat');
load('F:\master1\gxy_paper1\matlab代码\2878\X.mat');

%%  划分训练集和测试集
train_X = [X];
y = [labels];
xx = line_map(train_X);

nfolds = 2;
predict_y_all = y;
predict_y_all = 0;
crossval_idx = crossvalind('Kfold',y(:),nfolds);
RMSE_v = [];R_2=[];Adjusted_R=[];dim = size(xx,2);
ACC=[];SN=[];Spec=[];PE=[];NPV=[];F_score=[];MCC=[];
        
 for fold=1:nfolds

    %%%划分训练集和测试集
    train_idx = find(crossval_idx~=fold);
    test_idx = find(crossval_idx==fold);
    train_y = y(train_idx,1);
    train_x = xx(train_idx,:); 
    test_y = y(test_idx,1);
    test_x = xx(test_idx,:);
    
    %%  模型参数设置及训练模型
    %trees = 100; 原来% 决策树数目
    trees = 1; % 决策树数目
    %leaf  = 5; 原来% 最小叶子数
    leaf  = 1; % 最小叶子数
    OOBPrediction = 'on';  % 打开误差图
    OOBPredictorImportance = 'on'; % 计算特征重要性
    Method = 'classification';  % 选择回归或分类
    net = TreeBagger(trees, train_x, train_y, 'OOBPredictorImportance', OOBPredictorImportance,...
      'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
    importance = net.OOBPermutedPredictorDeltaError;  % 重要性
    
    %%  仿真测试
    [pyuce,scores] = predict(net, test_x);
    
    len = length(pyuce);
    data = zeros(len,1);
    for i=1:len
        data(i) = str2num(cell2mat(pyuce(i)));
    end
   
    ACC=[];SN=[];Spec=[];PE=[];NPV=[];F_score=[];MCC=[];AUC=[];
    [ACC_i,SN_i,Spec_i,PE_i,NPV_i,F_score_i,MCC_i] = roc2(data,test_y );
    
    [X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,scores(:,2),2);
    AUC=[AUC,AUC_i];ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];

 end

RF = plot(X_point,Y_point,'Color',[128 138 135]/255,'linewidth',2);
xlabel('False positive rate')
ylabel('true positive rate')
frame = getframe(gcf);
img=frame2im(frame);
imwrite(img,'F:\ROC.png');
hold off;
legend([MLapRVFL,SVM,RF],'MLapRVFL','SVM','RF')

