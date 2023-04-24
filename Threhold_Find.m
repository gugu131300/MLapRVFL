function [mcc_list,acc_lst,sn_list,sp_list,PE_list,F_list,rate_p_n_list] = Threhold_Find(score_s,label,min_Th,max_Th,step_Th)

max_label = max(label);
min_label = min(label);

mcc_list=[];acc_lst=[];sn_list=[];sp_list=[];rate_p_n_list=[];X=[];PE_list=[];F_list=[];
Threshold_t=0;
for k=min_Th:step_Th:max_Th
	Threshold_t = k;
	Predict_label = zeros(size(label,1),1);
	for i=1:size(label,1)
		SC = score_s(i,1);
		if SC>Threshold_t
			PP_ = max_label;
			Predict_label(i)=PP_;
		else
			PP_ = min_label;
			Predict_label(i)=PP_;
		end
	end
	ACC=[];
	MCC=[];
	SN=[];
	SPec=[];
	PE=[];
	NPV=[];
	F_Socre=[];
	[ACC,SN,SPec,PE,NPV,F_Socre,MCC] = roc(Predict_label,label);
	rate_s = sum(find(Predict_label==max_label))/sum(find(Predict_label==min_label));
	
	PE_list=[PE_list;PE];
	mcc_list=[mcc_list;MCC];
	acc_lst=[acc_lst;ACC];
	sn_list=[sn_list;SN];
	sp_list=[sp_list;SPec];
	rate_p_n_list=[rate_p_n_list;rate_s];
	F_list=[F_list;F_Socre];
	
	X=[X;k];
	

end
%mcc_list = mcc_list -0.01;
%PE_list = PE_list -0.01;
hold on
	plot(X,rate_p_n_list,'k','LineWidth',1.5);
	plot(X,F_list,'k','LineWidth',1.5);
    plot(X,PE_list,'c','LineWidth',1.5);
    
	plot(X,sn_list,'r','LineWidth',1.5);
	plot(X,sp_list,'g','LineWidth',1.5);
	plot(X,acc_lst,'y','LineWidth',1.5);
	plot(X,mcc_list,'b','LineWidth',1.5);
	
	grid on;ll=legend('SN', 'Spec', 'ACC', 'MCC');
	xlabel('Threshold');ylabel('Values');
	box on;
	grid off;
	set(get(gca,'XLabel'),'FontSize',18);
	set(get(gca,'YLabel'),'FontSize',18);
	set(gca,'FontSize',10);
	set(ll,'FontSize',10);
    
    frame = getframe(gcf);
    im = frame2im(frame);
    imwrite(im,'threshold.png');
    
end