function X_validate_proj = projection(inputValues, principal_comp, U , T)
if T==0
    X_validate = inputValues(:,50001:end)';
else
    X_validate = inputValues';
end

    
X_validatem = X_validate - repmat(mean(X_validate,1),size(X_validate,1),1);
%[U,~] = eig(X_validatem' * X_validatem);
%U = fliplr(U);
X_validate_proj = X_validatem * U(:,1:principal_comp);
end

