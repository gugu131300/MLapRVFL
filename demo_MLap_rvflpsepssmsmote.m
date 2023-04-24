clear
seed = 12345678;

load('F:\master1\gxy_paper1\matlab´úÂë\388\labels.mat');
load('F:\master1\gxy_paper1\matlab´úÂë\388\X.mat');
train_X = [X];
y = [labels];
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

    %fprintf("bbbbbbbbbbbbbbbbbbbb-results.Y")
   
    predict_y = results.Y;
    values = results.values;
    [ACC_i,SN_i,Spec_i,PE_i,NPV_i,F_score_i,MCC_i] = roc2(predict_y,test_y);
    score_s = 1./(1.0+exp(-1.0 * values));
    %[mcc_list,acc_lst,sn_list,sp_list] = Threhold_Findpsepssmsmote(score_s,test_y,0.1,0.9,0.0001);
    %[X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,score_s(:,1),1);
    ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];

 end

mean_acc=mean(ACC)
mean_spec=mean(Spec)
mean_mcc=mean(MCC)
mean_sn=mean(SN)
%mean_auc=mean(AUC)

