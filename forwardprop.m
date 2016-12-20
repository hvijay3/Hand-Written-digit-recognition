function [A1,A2,A3,output] = forwardprop(Lone,Parameter1,Parameter2)
 A1 = [1;Lone];
 A2 = [1;sigmg(Parameter1*A1)];
 
 %A2 = [1;SZ2];
 A3 = sigmg(Parameter2*A2);
% A3 =  sigmg(Z3);
 output = A3;
% output = sigmg(Parameter2*[1;sigmg(Parameter1*[1;Lone])]);
end
