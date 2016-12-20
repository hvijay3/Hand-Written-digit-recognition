clear all
cd('C:\Users\Harshit Vijayvargia\Documents\MATLAB\project3')
inputValues = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
inputValuesT = loadMNISTImages('t10k-images.idx3-ubyte');
labelsT = loadMNISTLabels('t10k-labels.idx1-ubyte');

X_training = inputValues(:,1:50000)';
X_projected = inputValues(:,50001:60000)';
X_trainingm = X_training - repmat(mean(X_training,1),size(X_training,1),1);
% PCA
[U, D] = eig(X_trainingm' * X_trainingm);
U = fliplr(U);
variance_per = (cumsum(flipud(diag(D)))./trace(D)).*100;
principal_comp = 65;
 X_training_proj = X_trainingm * U(:,1:principal_comp);
 X_validate_proj = projection(inputValues, principal_comp, U, 0);
 X_test_proj = projection(inputValuesT, principal_comp, U,1);

%X_training_proj = X_training;
%X_validate_proj = X_projected;



% Training network
tic

L1 = size(X_training_proj,2);
L2=80; 
L3=10;
H_layers = 1;    % L1,L2,L3 : size of input,hidden and output layer without bias
alpha =0.9;
sizeij = L2*(L1 +1 );
sizejk = L3* ( L2 + 1);
total_size= sizeij + sizejk ;     % Defining parameter vector size
%total_size = L2*(L1+1) + L3*(L2+1) + (H_layers-1)*L2*(L2+1);
thetaij = ((2/sqrt(principal_comp)) *  rand(sizeij,1)) - (1/sqrt(principal_comp));    % Initialising random weights
thetajk = ((2/sqrt(L2)) *  rand(sizejk,1)) - (1/sqrt(L2));    % Initialising random weights
theta = [thetaij;thetajk];
%Generating output Y with 10 classes
Y = labels; 
Y_10 = zeros(10,size(Y,1));
for output =1:size(Y,1)% output
    Y_10(Y(output)+1,output) =1;
end

Parameter1 = reshape(theta((1:L2*(L1+1)),:),[L2 L1+1]);                     % Generating weight matrices
Parameter2 = reshape(theta(L2*(L1+1)+1:L2*(L1+1)+L3*(L2+1),:),[L3 L2+1]);
stochastic_size = 1;
step_size = 0.5;
countx =1;
%D1old = zeros(size(Parameter1));
%D2old = zeros(size(Parameter2));
for iters = 50000;
X = X_training_proj;
batchsize = 1;

for iteration =1:iters
    %X = X_training_proj(randperm(size(X_training_proj,1),stochastic_size),:); % stochastic
    output_mat = zeros(10,size(X,1));        % Matrix for storing output
    Accum_par1 =zeros(L2,L1+1);              % Storing accumulated values after each batch is passed
    Accum_par2 = zeros(L3,L2+1);
    
    for i =1:batchsize         % 8 samples in training batch   :size(X,1)
        store = randperm(size(X,1),stochastic_size);
       Lone= X(store,:)';
        
        %Lone = X(i,:)';       % Lone, Ltwo, Lthree layers without bias
        %Ltwo = zeros(L2,1) ;
        %Lthree = zeros(L3,1);
        % Forward Propogation ,A1,A2,A3 layers with bias
        [A1,A2,A3,output] = forwardprop(Lone,Parameter1,Parameter2);  %Lone and Parameter1 and Parameter2 passed as arguement for
       % output_mat(:,i) = output;      % output stored
        %Ltwo = A2(2:size(A2,1),:);
        %Lthree = A3;
        %Backward Propogation
        e3 = output - Y_10(:,store) ;   % store added for stochastic
        %e3 = output - Y_10(:,i);
        error3(:,i) = e3;
        %error_training(iteration) = sum(error3.^2);
        [e2] = backward_prop(e3,Parameter2,A2);
        Accum_par1 = Accum_par1 + e2(2:size(e2, 1))*A1';
        Accum_par2 = Accum_par2 + e3*A2';
    end
    if iteration >1
        D1old = D1;
        D2old = D2;
    end
    % calculating D1 & D2 which are partial dervatives w.r.t cost function
     D1 = Accum_par1*(1/batchsize) ;   
     D2 = Accum_par2*(1/batchsize);
    
%     D1 = Accum_par1*(1/stochastic_size);   
%     D2 = Accum_par2*(1/stochastic_size);
    %mse
    %error_training(iteration) = mse(X_training_proj,Parameter1,Parameter2,Y_10,0);
    num = 50;
    if mod(iteration,num) == 0
   error_validate(iteration/num) = mse(X_validate_proj,Parameter1,Parameter2,Y_10,1);
   %error_training(iteration/num) = mse(X_training_proj(1:batchsize,:),Parameter1,Parameter2,Y_10,0);
    end
    %Updating parameters
    Parameter1 = Parameter1 - step_size.*D1;  % - step_size.*D1old.*alpha;
    Parameter2 = Parameter2 - step_size.*D2;   %-step_size.*D2old.*alpha;
end

[comparision,count] = classification (X_training_proj,Parameter1,Parameter2,labels,0);
[comparisionv,countv] = classification (X_validate_proj,Parameter1,Parameter2,labels,1);
[comparisiont,countt] = classification (X_test_proj,Parameter1,Parameter2,labelsT,0);

Accuracy (countx ,:) = [(countv/10000)*100,(count/50000)*100,(countt/10000)*100,toc];
tic;
countx= countx +1;
end
figure 
hold on
plot(error_training, 'r')
plot(error_validate,'g')