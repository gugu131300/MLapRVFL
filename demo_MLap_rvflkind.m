clear
seed = 12345678;

load('/seu_share/home/xiaopengfeng/230218819/data/psepssmsmote.mat');
load('/seu_share/home/xiaopengfeng/230218819/data/psepssmlabelsmote.mat');
%定义文件中使用到的数据集 
train_X = [psepssmsmote];
y = [psepssmlabelsmote];
%y(find(y==2)) = -1;
xx = line_map(train_X);

Lap_parameter.MaxIterations = 5;
Lap_parameter.gamma = 2^-1;
Lap_parameter.k_nn = 5;
Lap_parameter.ro = 1.5;
Lap_parameter.isMv = 1;

%option.ActivationFunction = radbas;
option.N = 100;
%option.bias = false;
%option.link = true;
%option.type_task = 2
option.n_center_vec = 140;
option.seed = 12345678;
option.lambda_1 = 0.01;
%option.lambda_2 = 0.01;
option.lambda_2 = 1;
option.sigma = 1;
option.type_task = 'binary classification';
option.lammda = 0.1;

nfolds = 10;
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
    %[mcc_list,acc_lst,sn_list,sp_list] = Threhold_Findpsepssmsmote(score_s,test_y,0.1,0.9,0.0001);
    [X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,score_s(:,1),1);
    ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];AUC=[AUC,AUC_i];
    
    break;

 end

hold on;
plot(X_point,Y_point,'Color',[255 97 0]/255,'linewidth',1,'DisplayName','PsePSSM');
legend
xlabel('False positive rate');ylabel('true positive rate');
frame = getframe(gcf);
img=frame2im(frame);
title('AUROC');
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%188D%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
seed = 12345678;
load('/seu_share/home/xiaopengfeng/230218819/data/hbgpcr.mat');
load('/seu_share/home/xiaopengfeng/230218819/data/labelgpcr.mat');
%定义文件中使用到的数据集 
train_X = [hbgpcr];
y = [labelgpcr];
%y(find(y==2)) = -1;
xx = line_map(train_X);

Lap_parameter.MaxIterations = 5;
Lap_parameter.gamma = 2^-1;
Lap_parameter.k_nn = 5;
Lap_parameter.ro = 1.5;
Lap_parameter.isMv = 1;

%option.ActivationFunction = radbas;
option.N = 100;
%option.bias = false;
%option.link = true;
%option.type_task = 2
option.n_center_vec = 140;
option.seed = 12345678;
option.lambda_1 = 0.01;
%option.lambda_2 = 0.01;
option.lambda_2 = 1;
option.sigma = 1;
option.type_task = 'binary classification';
option.lammda = 0.1;

nfolds = 10;
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
    %[mcc_list,acc_lst,sn_list,sp_list] = Threhold_Findpsepssmsmote(score_s,test_y,0.1,0.9,0.0001);
    [X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,score_s(:,1),1);
    ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];AUC=[AUC,AUC_i];
    
    break;

 end

plot(X_point,Y_point,'Color',[175 81 108]/255,'linewidth',1,'DisplayName','188D');
legend
xlabel('False positive rate');ylabel('true positive rate');
frame = getframe(gcf);
img=frame2im(frame);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%CTDC%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
seed = 12345678;

load('/seu_share/home/xiaopengfeng/230218819/data/napCTDC.mat');
load('/seu_share/home/xiaopengfeng/230218819/data/Labels.mat');
%定义文件中使用到的数据集 
train_X = [napCTDC];
y = [label];
%y(find(y==2)) = -1;
xx = line_map(train_X);

Lap_parameter.MaxIterations = 5;
Lap_parameter.gamma = 2^-1;
Lap_parameter.k_nn = 5;
Lap_parameter.ro = 1.5;
Lap_parameter.isMv = 1;

%option.ActivationFunction = radbas;
option.N = 100;
%option.bias = false;
%option.link = true;
%option.type_task = 2
option.n_center_vec = 140;
option.seed = 12345678;
option.lambda_1 = 0.01;
%option.lambda_2 = 0.01;
option.lambda_2 = 1;
option.sigma = 1;
option.type_task = 'binary classification';
option.lammda = 0.1;

nfolds = 10;
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
    %[mcc_list,acc_lst,sn_list,sp_list] = Threhold_Findpsepssmsmote(score_s,test_y,0.1,0.9,0.0001);
    [X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,score_s(:,1),1);
    ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];AUC=[AUC,AUC_i];
    
    break;

 end
 
plot(X_point,Y_point,'Color',[209 223 33]/255,'linewidth',1,'DisplayName','CTDC');
legend
xlabel('False positive rate');ylabel('true positive rate');
frame = getframe(gcf);
img=frame2im(frame);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%CKSAAP%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
seed = 12345678;
load('/seu_share/home/xiaopengfeng/230218819/data/napCKSAAP.mat');
load('/seu_share/home/xiaopengfeng/230218819/data/Labels.mat');
%定义文件中使用到的数据集 
train_X = [napCKSAAP];
y = [label];
%y(find(y==2)) = -1;
xx = line_map(train_X);

Lap_parameter.MaxIterations = 5;
Lap_parameter.gamma = 2^-1;
Lap_parameter.k_nn = 5;
Lap_parameter.ro = 1.5;
Lap_parameter.isMv = 1;

%option.ActivationFunction = radbas;
option.N = 100;
%option.bias = false;
%option.link = true;
%option.type_task = 2
option.n_center_vec = 140;
option.seed = 12345678;
option.lambda_1 = 0.01;
%option.lambda_2 = 0.01;
option.lambda_2 = 1;
option.sigma = 1;
option.type_task = 'binary classification';
option.lammda = 0.1;

nfolds = 10;
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
    %[mcc_list,acc_lst,sn_list,sp_list] = Threhold_Findpsepssmsmote(score_s,test_y,0.1,0.9,0.0001);
    [X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,score_s(:,1),1);
    ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];AUC=[AUC,AUC_i];
    
    break;

 end

plot(X_point,Y_point,'Color',[100 156 107]/255,'linewidth',1,'DisplayName','CKSAAP');
legend
xlabel('False positive rate');ylabel('true positive rate');
frame = getframe(gcf);
img=frame2im(frame);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%QSOrder%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
seed = 12345678;

load('/seu_share/home/xiaopengfeng/230218819/data/napQSOrder.mat');
load('/seu_share/home/xiaopengfeng/230218819/data/Labels.mat');
%定义文件中使用到的数据集 
train_X = [napQSOrder];
y = [label];
%y(find(y==2)) = -1;
xx = line_map(train_X);

Lap_parameter.MaxIterations = 5;
Lap_parameter.gamma = 2^-1;
Lap_parameter.k_nn = 5;
Lap_parameter.ro = 1.5;
Lap_parameter.isMv = 1;

%option.ActivationFunction = radbas;
option.N = 100;
%option.bias = false;
%option.link = true;
%option.type_task = 2
option.n_center_vec = 140;
option.seed = 12345678;
option.lambda_1 = 0.01;
%option.lambda_2 = 0.01;
option.lambda_2 = 1;
option.sigma = 1;
option.type_task = 'binary classification';
option.lammda = 0.1;

nfolds = 10;
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
    %[mcc_list,acc_lst,sn_list,sp_list] = Threhold_Findpsepssmsmote(score_s,test_y,0.1,0.9,0.0001);
    [X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,score_s(:,1),1);
    ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];AUC=[AUC,AUC_i];
    
    break;

 end

plot(X_point,Y_point,'Color',[153 41 215]/255,'linewidth',1,'DisplayName','QSOrder');%%%%%
legend
xlabel('False positive rate');ylabel('true positive rate');
frame = getframe(gcf);
img=frame2im(frame);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%AAC%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
seed = 12345678;

load('/seu_share/home/xiaopengfeng/230218819/data/napAAC.mat');
load('/seu_share/home/xiaopengfeng/230218819/data/Labels.mat');
%定义文件中使用到的数据集 
train_X = [napAAC];
y = [label];
%y(find(y==2)) = -1;
xx = line_map(train_X);

Lap_parameter.MaxIterations = 5;
Lap_parameter.gamma = 2^-1;
Lap_parameter.k_nn = 5;
Lap_parameter.ro = 1.5;
Lap_parameter.isMv = 1;

%option.ActivationFunction = radbas;
option.N = 100;
%option.bias = false;
%option.link = true;
%option.type_task = 2
option.n_center_vec = 140;
option.seed = 12345678;
option.lambda_1 = 0.01;
%option.lambda_2 = 0.01;
option.lambda_2 = 1;
option.sigma = 1;
option.type_task = 'binary classification';
option.lammda = 0.1;

nfolds = 10;
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
    %[mcc_list,acc_lst,sn_list,sp_list] = Threhold_Findpsepssmsmote(score_s,test_y,0.1,0.9,0.0001);
    [X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,score_s(:,1),1);
    ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];AUC=[AUC,AUC_i];
    
    break;

 end

plot(X_point,Y_point,'Color',[149 150 207]/255,'linewidth',1,'DisplayName','AAC');%%%
legend
xlabel('False positive rate');ylabel('true positive rate');
frame = getframe(gcf);
img=frame2im(frame);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%CTDD%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
seed = 12345678;

load('/seu_share/home/xiaopengfeng/230218819/data/napCTDD.mat');
load('/seu_share/home/xiaopengfeng/230218819/data/Labels.mat');
%定义文件中使用到的数据集 
train_X = [napCTDD];
y = [label];
%y(find(y==2)) = -1;
xx = line_map(train_X);

Lap_parameter.MaxIterations = 5;
Lap_parameter.gamma = 2^-1;
Lap_parameter.k_nn = 5;
Lap_parameter.ro = 1.5;
Lap_parameter.isMv = 1;

%option.ActivationFunction = radbas;
option.N = 100;
%option.bias = false;
%option.link = true;
%option.type_task = 2
option.n_center_vec = 140;
option.seed = 12345678;
option.lambda_1 = 0.01;
%option.lambda_2 = 0.01;
option.lambda_2 = 1;
option.sigma = 1;
option.type_task = 'binary classification';
option.lammda = 0.1;

nfolds = 10;
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
    %[mcc_list,acc_lst,sn_list,sp_list] = Threhold_Findpsepssmsmote(score_s,test_y,0.1,0.9,0.0001);
    [X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,score_s(:,1),1);
    ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];AUC=[AUC,AUC_i];
    
    break;

 end

plot(X_point,Y_point,'Color',[247 250 106]/255,'linewidth',1,'DisplayName','CTDD');%%%
legend
xlabel('False positive rate');ylabel('true positive rate');
frame = getframe(gcf);
img=frame2im(frame);
imwrite(img,'ROCkind.png');
hold off;