function [ACC,SN,Spec,PE,NPV,F_score,MCC,TruePositive2,PE2] = roc( predict_label,test_data_label )
%ROC Summary of this function goes here
%   Detailed explanation goes here
l=length(predict_label);
TruePositive = 0;
TrueNegative = 0;
FalsePositive = 0;
FalseNegative = 0;
max_label = max(test_data_label);
min_label = min(test_data_label);
TruePositive2 = [];
FalsePositive2 = [];
for k=1:l
    if test_data_label(k)==max_label & predict_label(k)==max_label  %������
        TruePositive = TruePositive +1;
    end
    if test_data_label(k)==min_label & predict_label(k)==min_label %������
        TrueNegative = TrueNegative +1;
    end 
    if test_data_label(k)==min_label & predict_label(k)==max_label  %������
        FalsePositive = FalsePositive +1;
    end

    if test_data_label(k)==max_label & predict_label(k)==min_label  %������
        FalseNegative = FalseNegative +1;
    end
    TruePositive2(k) = TruePositive;
    FalsePositive2(k) = FalsePositive;
end
%TruePositive
%TrueNegative
%FalsePositive
%FalseNegative
ACC = (TruePositive+TrueNegative)./(TruePositive+TrueNegative+FalsePositive+FalseNegative);
SN = TruePositive./(TruePositive+FalseNegative);
Spec = TrueNegative./(TrueNegative+FalsePositive);
PE = TruePositive./(TruePositive+FalsePositive);
NPV = TrueNegative./(TrueNegative+FalseNegative);
F_score = 2*(SN*PE)./(SN+PE);
MCC = (TruePositive*TrueNegative-FalsePositive*FalseNegative)./sqrt(  (TruePositive+FalseNegative)...
    *(TrueNegative+FalsePositive)*(TruePositive+FalsePositive)*(TrueNegative+FalseNegative));

PE2= [];
for i=1:l
    PE2(i) = TruePositive2(i)./(TruePositive2(i)+FalsePositive2(i));
end

end