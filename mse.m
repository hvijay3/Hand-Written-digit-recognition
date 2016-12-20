function error = mse(X_training_proj,Parameter1,Parameter2,Y_10,v)
for i =1:size(X_training_proj,1)
    [~,~,~,output] = forwardprop(X_training_proj(i,:)',Parameter1,Parameter2);
    output_mat1(:,i) = output;
end

if v==1
    error = (1/size(X_training_proj,1)).* sum(sum((Y_10(:,50001:end) - output_mat1).^2));
else
    error = (1/size(X_training_proj,1)).* sum(sum((Y_10(:,1:size(X_training_proj,1)) - output_mat1).^2));
end
end
