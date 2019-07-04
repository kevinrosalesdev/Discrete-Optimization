include "globals.mzn";
int: n;
array[1..n, 1..n] of var 1..n*n: cuadrado;
var int: suma;

%Restricciones -> sumas de diagonales, filas y columnas = valor
constraint forall(i in 1..n)(sum(j in 1..n)(cuadrado[i,j]) = suma);
constraint forall(i in 1..n)(sum(j in 1..n)(cuadrado[j,i]) = suma);
constraint sum(i in 1..n)(cuadrado[i,i]) = suma;
constraint sum(i in 1..n)(cuadrado[i,n+1-i]) = suma;

%Ruptura de simetrías en las soluciones
constraint lex_lesseq([cuadrado[i,j]| i,j in 1..n], [cuadrado[j,i]| i,j in 1..n]);
constraint lex_lesseq([cuadrado[n+1-i,j]| i,j in 1..n], [cuadrado[n+1-j,i]| i,j in 1..n]);
constraint lex_lesseq([cuadrado[n+1-i,j]| i,j in 1..n], [cuadrado[j,i]| i,j in 1..n]);

%Restricción -> todos los valores son distintos
constraint all_different(cuadrado);

solve satisfy;