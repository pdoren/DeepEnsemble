%% Funcion para obtener el error de prediccion cuadratico medio (MSPE) y el 
% error de prediccion cuadratico medio normalizado (NMSPE) de una serie de 
% tiempo
%
% [e_mspe,e_nmspe] = calc_error_time_series(y,t)
% -> y       : Vector con datos de prediccion.
% -> t       : Vector con valores reales de la serie.
% <- e_mspe  : Valor del error de prediccion cuadratico medio entre "t" e "y".
% <- e_nmspe : Valor del error de prediccion cuadratico medio normalizado entre "t" e "y".

% Jorge Vergara
% 2011/Nov

function [e_mspe,e_nmspe] = calc_error_time_series(y,t)
e = y-t;
dontcares = find(~isfinite(e));
e(dontcares) = 0;
numerator = sum(sum(e.^2));
denominator = sum(sum((y-mean(y)).^2));
numElements = prod(size(e)) - length(dontcares);
if (numElements == 0)
  e_mspe = 0;
  e_nmspe = 0;
else
  e_mspe  = numerator / numElements;
  e_nmspe = numerator / denominator;
end

end
