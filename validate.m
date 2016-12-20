function X_validate_proj = projection(inputValues, principal_comp, U)
X_validate = inputValues(:,50001:end)';
X_validatem = X_validate - repmat(mean(X_validate,1),size(X_validate,1),1);
X_validate_proj = X_validatem * U(:,1:principal_comp);
end

