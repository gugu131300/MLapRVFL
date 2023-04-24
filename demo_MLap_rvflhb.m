clear
seed = 12345678;

load('F:\master1\gxy_paper1\matlab代码\388\X.mat');
load('F:\master1\gxy_paper1\matlab代码\388\labels.mat');

train_X = [X];
y = [labels];
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
    [X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,score_s(:,1),1);
    ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];AUC=[AUC,AUC_i];
    
    break;

 end

hold on;
plot(X_point,Y_point,'Color',[4 157 107]/255,'linewidth',2);
xlabel('False positive rate');ylabel('true positive rate');
frame = getframe(gcf);
%set(gcf,'Position',[300 300 400 400]);%消除白边
%set(gca,'Position',[0 0 1 1]);%消除白边
img=frame2im(frame);
imwrite(img,'F:\ROC222.png');

title('在一幅图中绘制多条曲线');
%imshow(img,'Border','tight');

mean_acc=mean(ACC)
mean_spec=mean(Spec)
mean_mcc=mean(MCC)
mean_sn=mean(SN)
mean_auc=mean(AUC)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('F:\master1\gxy_paper1\matlab代码\388\X.mat');
load('F:\master1\gxy_paper1\matlab代码\388\labels.mat');

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
option.lambda_1 = 0.01;
option.lambda_2 = 1;
option.sigma = 1;
option.type_task = 'binary classification';
option.lammda = 0.1;

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
    [X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,score_s(:,1),1);
    ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];AUC=[AUC,AUC_i];
    
    break;

 end

plot(X_point,Y_point,'Color',[175 81 108]/255,'linewidth',2);
hold on;
xlabel('False positive rate');ylabel('true positive rate');
frame = getframe(gcf);
img=frame2im(frame);

legend('sin(x/2)','2016')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
seed = 12345678;

load('F:\master1\gxy_paper1\matlab代码\388\X.mat');
load('F:\master1\gxy_paper1\matlab代码\388\labels.mat');

train_X = [X];
y = [labels];
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
    [X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,score_s(:,1),1);
    ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];AUC=[AUC,AUC_i];
    
    break;

 end

plot(X_point,Y_point,'Color',[128 138 135]/255,'linewidth',2);
xlabel('False positive rate');ylabel('true positive rate');
frame = getframe(gcf);
img=frame2im(frame);
legend('sin(x/2)','2016')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
seed = 12345678;

load('F:\master1\gxy_paper1\matlab代码\388\X.mat');
load('F:\master1\gxy_paper1\matlab代码\388\labels.mat');

train_X = [X];
y = [labels];
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
    [X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,score_s(:,1),1);
    ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];AUC=[AUC,AUC_i];
    
    break;

 end

plot(X_point,Y_point,'Color',[65 105 225]/255,'linewidth',2);
xlabel('False positive rate');ylabel('true positive rate');
frame = getframe(gcf);
img=frame2im(frame);
imwrite(img,'F:\ROC222.png');

%title('在一幅图中绘制多条曲线');
%imshow(img,'Border','tight');
hold off;
legend('sin(x/2)','2016')
