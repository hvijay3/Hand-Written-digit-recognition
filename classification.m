function [comparision,count] = classification (X_training_proj,Parameter1,Parameter2,labels,v)
for i =1:size(X_training_proj,1)
    [~,~,~,output] = forwardprop(X_training_proj(i,:)',Parameter1,Parameter2);
    output_mat1(:,i) = output;
end
[M, I] = max(output_mat1);
if v==1
    comparision = [labels(50000+1:50000+size(X_training_proj,1)),I'-ones(size(X_training_proj,1),1)];
else
    comparision = [labels(1:size(X_training_proj,1)),I'-ones(size(X_training_proj,1),1)];

end
count = 0;
for c= 1:size(X_training_proj,1)
    if comparision(c,1)==comparision(c,2);
        count= count+1;
    end
end
end