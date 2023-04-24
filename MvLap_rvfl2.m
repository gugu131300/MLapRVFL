function [predict_results] = MvLap_rvfl2(trainX,trainY,testX,testY,option,Lap_parameter)

% Option.N :      number of hidden neurons
% Option.bias:    whether to have bias in the output neurons
% option.link:    whether to have the direct link.
% option.ActivationFunction:Activation Functions used.   
% option.seed:    Random Seeds
% option,mode     1: regularized least square, 2: Moore-Penrose pseudoinverse
%option.lambda_1d
%option.lambda_2
if ~isfield(option,'N')|| isempty(option.N)
    option.N=100;
end
if ~isfield(option,'bias')|| isempty(option.bias)
    option.bias=false;
end
if ~isfield(option,'link')|| isempty(option.link)
    option.link=true;
end
if ~isfield(option,'ActivationFunction')|| isempty(option.ActivationFunction)
    option.ActivationFunction='radbas';
end
%random seed
if ~isfield(option,'seed')|| isempty(option.seed)
    option.seed=0;
end

randn('state',option.seed);

obj_v=[];

%trainY_temp=trainY;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%for task ,dyj1
switch lower(option.type_task)
    case {'regression','binary classification'} 
			trainY_temp = trainY;
    case {'classification'} 
       [ trainY_temp ] = y2ooh( trainY, length(unique(trainY)) );
end

[Nsample,Nfea]=size(trainX);
N=option.N;

%init the weights of network via Gaussian distribution 
Weight=randn(Nfea,N);
Bias=randn(1,N); 

Bias_train=repmat(Bias,Nsample,1);   %%%%% Nsample*N
H=trainX*Weight+Bias_train;          %%%%% Nsample*2N
b = Weight

%the Activation Function. sigmoid or rbf

switch lower(option.ActivationFunction)
    case {'sig','sigmoid'}
      
        H = 1 ./ (1 + exp(-H));
    case {'radbas'}
       
        H = radbas(H);
           
end

if option.bias
   H=[H,ones(Nsample,1)]; 
end
if option.link
 
   H=[H,trainX];    
     
end
H(isnan(H))=0;

H_b = size(H);    %%%%%%%%%% 0*288

L_train=[];
%D = diag(mu);
	
		W1 = kernel_RBF(H,H,Lap_parameter.gamma);
		sn = size(W1,1);    
		S_1 =preprocess_PNN(W1,Lap_parameter.k_nn);
		d_1 = sum(S_1);
		D_1 = diag(d_1);
		L_D_1 = D_1 - S_1;
		d_tmep_1=eye(sn)/(D_1^(1/2));
		L1 = d_tmep_1*L_D_1*d_tmep_1;
		L_train(:,:,1)=L1;
		
		W2 = kernel_cos(H,H);
		W2 = Knormalized(W2);
		S_2 =preprocess_PNN(W2,Lap_parameter.k_nn);
		d_2 = sum(S_2);
		D_2 = diag(d_2);
		L_D_2 = D_2 - S_2;
		d_tmep_2=eye(sn)/(D_2^(1/2));
		L2 = d_tmep_2*L_D_2*d_tmep_2;
		L_train(:,:,2)=L2;
		
		W3 = kernel_Polynomial(H,H,Lap_parameter.gamma);
		W3 = Knormalized(W3);
		S_3 =preprocess_PNN(W3,Lap_parameter.k_nn);
		d_3 = sum(S_3);
		D_3 = diag(d_3);
		L_D_3 = D_3 - S_3;
		d_tmep_3=eye(sn)/(D_3^(1/2));
		L3 = d_tmep_3*L_D_3*d_tmep_3;
		L_train(:,:,3)=L3;
		
		W4 = kernel_sigmoid(H,H,Lap_parameter.gamma);
		W4 = Knormalized(W4);
		S_4 = preprocess_PNN(W4,Lap_parameter.k_nn);
		d_4 = sum(S_4);
		D_4 = diag(d_4);
		L_D_4 = D_4 - S_4;
		d_tmep_4 = eye(sn)/(D_4^(1/2));
		L4 = d_tmep_4*L_D_4*d_tmep_4;
		L_train(:,:,4)=L4;
	
	if Lap_parameter.isMv ==1
			weights = ones(4,1)*0.25;
			
		elseif Lap_parameter.isMv ==0%mean weight
			weights = ones(4,1)*0.25;
		elseif Lap_parameter.isMv ==2%use RBF
			weights = [1;0;0;0];
		elseif Lap_parameter.isMv ==3%use cos
			weights = [0;1;0;0];
		elseif Lap_parameter.isMv ==4%use Polynomial
			weights =[0;0;1;0];;
		elseif Lap_parameter.isMv ==5%use sigmoid
			weights = [0;0;0;1];;
    		end
	 
    C = option.lambda_1;
    if N<Nsample
     beta=(eye(size(H,2))*C+H' * H) \ H'*trainY_temp;  
     beta1 = size(beta);
    else
     beta=H'*((eye(size(H,1))*C+H* H') \ trainY_temp); 
     beta2 = size(beta);                               
    end
    
	 D = zeros(size(beta,1),size(beta,1));   
     
     
	for i = 1:Lap_parameter.MaxIterations
		zx = 2*beta.*beta; 
		D = zx.^(-0.5);
		D = diag(zx);
        
        D_a = size(D);               
        H_a = size(H' * H);           
        
		L = combine_Ls(weights, L_train);
        
        L_a = size(L);             
		beta=(H' * H + option.lambda_2*H'*L*H+ D*C) \ H'*trainY_temp; 
        beta_a = size(beta);         
		
		F = H*beta;
		if Lap_parameter.isMv ==1
			weights=computing_weights(F,L_train,Lap_parameter.ro,1);
			
		elseif Lap_parameter.isMv ==0 %mean weight
			weights = ones(4,1)*0.25;
		elseif Lap_parameter.isMv ==2 %use RBF
			weights = [1;0;0;0];
		elseif Lap_parameter.isMv ==3 %use cos
			weights = [0;1;0;0];
		elseif Lap_parameter.isMv ==4 %use Polynomial
			weights =[0;0;1;0];;
		elseif Lap_parameter.isMv ==5 %use sigmoid
			weights = [0;0;0;1];;
		end
		
	end


beta(find(abs(beta)<0.01))=0;
Bias_test = repmat(Bias,numel(testY),1);
H_test = testX*Weight+Bias_test;
a = Weight


switch lower(option.ActivationFunction)
   case {'sig','sigmoid'}
       H_test = 1 ./ (1 + exp(-H_test));
	case {'radbas'}
        H_test = radbas(H_test);
end

if option.bias
   H_test=[H_test,ones(numel(testY),1)]; 
end

if option.link

  H_test=[H_test,testX]; 
       
end
H_test(isnan(H_test))=0;
H_test_a = size(H_test);                    
testY_temp=H_test*beta;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%dyj2
switch lower(option.type_task)
    case {'regression'}
        predict_results.Y = testY_temp;
    case {'classification'}
       	[~, test_max_index] = max(testY_temp');
		 predict_results.Y = test_max_index';
	case {'binary classification'}
			predict_results.Y = sign(testY_temp);
end

predict_results.values = testY_temp;
predict_results.Y = sign(testY_temp);
predict_results.beta=beta;
predict_results.weights = weights;

end

function k = kernel_RBF(X,Y,gamma)
	r2 = repmat( sum(X.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(X,1) )' ...
	- 2*X*Y' ;
	k = exp(-r2*gamma); % RBF核矩�?
end

function W = kernel_cos(X,Y)
	n_obi=size(X,1);
	n_obj=size(Y,1);
	W = zeros(n_obi,n_obj);
	for i=1:n_obi
		for j=i:n_obj
				Profile1 = X(i,:);
				Profile2 = Y(j,:);
				sim_v=kernel_to_sim_cos(Profile1,Profile2);
				W(i,j) = sim_v;
				W(j,i) = sim_v;
		end
	end
	
	
end

%cosine similiarity 
function sim_cos=kernel_to_sim_cos(v1,v2)
 
	sim_cos = dot(v1,v2)/(norm(v1)*norm(v2));

end

function S=preprocess_PNN(S,p)
%preprocess_PNN sparsifies the similarity matrix S by keeping, for each
%drug/target, the p nearest neighbors and discarding the rest.
%
% S = preprocess_PNN(S,p)

    NN_mat = zeros(size(S));

    % for each drug/target...
    for j=1:length(NN_mat)
        row = S(j,:);                           % get row corresponding to current drug/target
        row(j) = 0;                             % ignore self-similarity
        [~,indx] = sort(row,'descend');         % sort similarities descendingly
        indx = indx(1:p);                       % keep p NNs
        NN_mat(j,indx) = S(j,indx);             % keep similarities to p NNs
        NN_mat(j,j) = S(j,j);                   % also keep the self-similarity (typically 1)
    end

    % symmetrize the modified similarity matrix
    S = (NN_mat+NN_mat')/2;

end

function k = kernel_Polynomial(X,Y,gamma)
	coef0 = 0.01;
	d =2.2;
	k = (gamma*X*Y' + coef0).^d; %核矩�?
end

%sigmoid kernel
function k = kernel_sigmoid(X,Y,gamma)
	coef0=0.01;
	
	k = tanh(gamma*X*Y' + coef0); %核矩�?
end

function result = combine_Ls(weights, kernels)
    % length of weights should be equal to length of matrices
    n = length(weights);
    result = zeros(size(kernels(:,:,1)));    
    
    for i=1:n
        result = result + weights(i) * kernels(:,:,i);
    end
end

function weights=computing_weights(F,L_l,gamma,dim)

w = zeros(size(L_l,3),1);
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

function S=Knormalized(K)
%kernel normilization
K = abs(K);
kk = K(:);
kk(find(kk==0)) = [];
min_v = min(kk);
K(find(K==0))=min_v;

D=diag(K);
D=sqrt(D);
S=K./(D*D');

end