function [predict_label,score_s,model] = fis_ssr(X_train,Y_train,X_test,options)

% X_train: n_tr * n_features
% Y_train: n_tr * c
% X_test: n_te * n_features
% Y_test: n_te * c
% options:
% options.lambda: % regularization parameter for ridge regression TSK
%options.beta
%options.ga
% options.k: number of fuzzy rules
% options.h: adjustable parameter for fcm used for generating antecedent
%options.ro    parameters.
%options.maxT

seed = 12345678;
%rand('seed', seed);
y_pred_test=[];
[n_tr,dims] = size(X_train);
%X_uni = [X_train',X_test']; 
%X_uni = X_uni*diag(sparse(1./sqrt(sum(X_uni.^2))));
%X_uni_train = X_uni(:,1:n_tr)';
%X_uni_test = X_uni(:,(n_tr+1):end)';
D_train = X_train;
D_test = X_test;
model =[];
obj_list = [];

[v_train,b_train] = gene_ante_fcm(D_train,options);
[v_test,b_test] = gene_ante_fcm(D_test,options);

G_train = calc_x_g(D_train,v_train,b_train);
G_test = calc_x_g(D_test,v_test,b_test);

clear D_train;
clear D_test;
clear X_train;
clear X_test;
clear v_train;
clear b_train;
clear v_test;
clear b_test;

uniqlabels=unique(Y_train);

num_classes = length(uniqlabels);
[ data_y_ooh_train ] = y2ooh( Y_train, num_classes );
dims1 = dims + 1;
Whole_dims = options.k*(dims1);

G = rand(Whole_dims,num_classes); %%%%%%%%%%%这里是P

S_list = zeros(Whole_dims,Whole_dims,num_classes);  %%%%%%%%%%%这里是S

		for j=1:num_classes
			label_i = find(Y_train==j);
			%p_i = length(label_i)/n_tr;
			SS_i = cov(G_train(label_i,:));%%%%%%%%%G_train中第label_i行得出的数组相乘
			
			S_list(:,:,j) = SS_i;
		end
		
weights_1 = ones(num_classes,1);
weights_1(1:num_classes) = 1/num_classes;

	for t = 1:options.maxT
		obj_v = computing_err(data_y_ooh_train,G_train,G);
		obj_list = [obj_list,obj_v];
		S_F  = combine_s(weights_1.^options.ro, S_list);
		%AX_Y = G_train*G - data_y_ooh_train;
		W = computer_W(G);
		D = computer_D(G,num_classes,options.k,dims1);
		for j=1:num_classes
			D_j = diag(D(:,j));
			A=(G_train'*G_train + options.beta*S_F + 4*options.lambda*W + options.ga*D_j);
			g_j = (A)\(G_train'*data_y_ooh_train(:,j));
			G(:,j) = g_j;
		end

		weights_1=computing_weights(G,S_list,options.ro,1);
		
	end

model.options = options;
eps = 0.01;
G(find(abs(G)<=eps))=0;
model.G = G;
model.weights = weights_1;
model.S_list = S_list;
model.errs = obj_list;
y_pred_test = G_test*G;
%sum_s = sum(y_pred_test');
%score_s = zeros(size(X_test,1),num_classes);
%sum_s = sum_s';
%for i=1:num_classes
%	score_s(:,i) = y_pred_test(:,i)./sum_s;
%end
score_s = y_pred_test;
[~, test_max_index] = max(y_pred_test');
predict_label = test_max_index';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function W = computer_W(G)

	W = zeros(size(G,1),size(G,1));

	for s=1:size(G,1)
		w = norm(G(s,:), 2);
		w = w^(1.5);
		if w==0
			w=0.01;
		end
		W(s,s) = 1./(4*w);
		%W(s,s) = 0.25*w;
        
	end

end

function [v,b] = gene_ante_fcm(data,options)
% 2018-04-23 PengXu x1724477385@126.com
% Generate the antecedents parameters of TSK FS by FCM

% data: n_example * n_features
% options.k: the number of rules
% options.h: the adjustable parameter of kernel width 
% of Gaussian membership function.

% return::v: the clustering centers -- k * n_features
% return::b: kernel width of corresponding clustering centers

k = options.k;
h = options.h;
[n_examples, d] = size(data);
% options: exponent for partition matrix & iterations & threshold & display
[v,U,~] = fcm(data,k,[2,NaN,1.0e-6,0]);

for i=1:k
    v1 = repmat(v(i,:),n_examples,1);
    u = U(i,:);
    uu = repmat(u',1,d);
    b(i,:) = sum((data-v1).^2.*uu,1)./sum(uu)./1;
end
b = b*h+eps;


end

function [x_g] = calc_x_g(x,v,b)
% 2018-04-23 PengXu x1724477385@126.com
% Calculate the X_g by x * fire-level

% x: the original data -- n_examples * n_features
% v: clustering centers of the fuzzy rule base -- k * n_features
% b: kernel width of the corresponding centers of the fuzzy rule base
% x_g: data in the new fuzzy feature space -- n_examples * (n_features+1)k

n_examples = size(x,1);
x_e = [x,ones(n_examples,1)];
[k,d] = size(v); % k: number of rules of TSK; d: number of dimensions

for i=1:k
    v1 = repmat(v(i,:),n_examples,1);
    bb = repmat(b(i,:),n_examples,1);
    wt(:,i) = exp(-sum((x-v1).^2./bb,2));
end

wt2 = sum(wt,2);

% To avoid the situation that zeros are exist in the matrix wt2
ss = wt2==0;
wt2(ss,:) = eps;
wt = wt./repmat(wt2,1,k);

x_g = [];
for i=1:k
    wt1 = wt(:,i);
    wt2 = repmat(wt1,1,d+1);
    x_g = [x_g,x_e.*wt2];
end

end

function [ data_y_ooh ] = y2ooh( y_label, num_classes )
% Transform the labels to the form of 'one-of-hot'
n_examples = size(y_label, 1);
data_y_ooh = zeros(n_examples, num_classes);
for i=1:n_examples
    index = y_label(i, :);
    data_y_ooh(i, index) = 1;
end
end

function D = computer_D(G,num_classes,num_fuzzy_sets,dims1)

D = zeros(dims1*num_fuzzy_sets,num_classes);
	for i=1:num_classes
		V=[];
		for j=1:num_fuzzy_sets
			end_i = j*dims1;
		    start_i = j*dims1 - (dims1 - 1);
			v = G(start_i:end_i,i);
			v = 2*norm(v, 2);
			if v==0
				v = 0.01;
			end
			v = 1/v;
			V = v*ones(dims1,1);
		end
		D(start_i:end_i,i) = V;
	end


end

function weights=computing_weights(F,L_l,gamma,dim) %%%%%F是P

w = zeros(size(L_l,3),1);%%%%%% L-1是S_list
weights = w;
e = 1/(gamma - 1);
	for i=1:length(w)   
		if dim ==1
			d = F'*L_l(:,:,i)*F;
		else
			d = F*L_l(:,:,i)*F';
		end
		s = (1/trace(d))^e;
		w(i) = s;
	end
	for i=1:length(w)
		weights(i) = w(i)/(sum(w));
	end

end

function result = combine_s(weights, kernels)
    % length of weights should be equal to length of matrices
    n = length(weights);
    result = zeros(size(kernels(:,:,1)));    
    
    for i=1:n
        result = result + weights(i) * kernels(:,:,i);
    end
end

function obj_v = computing_err(y,X,A)
		obj_1 = y-X*A;
	obj_v = norm(obj_1,'fro') ;

end

function s=sigmod_f(H)


s = 1 ./ (1 + exp(-H));

end