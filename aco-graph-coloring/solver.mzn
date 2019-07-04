include "globals.mzn";

int: nSize;
set of int: size = 1..nSize;

int : nRestrictions;
set of int: nrestrictions = 1..nRestrictions;

array[size] of int : countries;
array[size] of var 1..nSize : coloresPaises;


array[nrestrictions] of int : A;
array[nrestrictions] of int : B;

constraint coloresPaises[1] = 1;

% Usamos value_precede como restricción lexicográfica, obligando a que dentro de 'coloresPaises', 
% antes de aparecer un nuevo color deben haber aparecido todos los de detrás.
constraint forall(i in 1..nSize-1)(value_precede(i,i+1,coloresPaises));

constraint forall(i in nrestrictions) (coloresPaises[A[i]] != coloresPaises[B[i]]);

solve minimize(max(coloresPaises));
