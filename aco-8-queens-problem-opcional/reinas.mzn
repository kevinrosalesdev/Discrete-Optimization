array [1..8] of var 1..8 : column;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Solución Planteada por nuestro grupo (Funcional pero más costosa):
%
% constraint forall(i in 1..8, j in 1..8 where i!=j) (column[i]!=column[j]); % en la misma fila
%
% constraint forall(i in 1..8, j in 1..8 where (i+j<9 /\ column[i]+j<9)) (column[i+j]!=column[i]+j); %d arriba
% constraint forall(i in 1..8, j in 1..8 where (i+j<9 /\ column[i]-j>0)) (column[i+j]!=column[i]-j); %d abajo
% 
% constraint forall(i in 1..8, j in 1..8 where (j<i /\ column[i]+j<9)) (column[i-j]!=column[i]+j); %i arriba
% constraint forall(i in 1..8, j in 1..8 where (j<i /\ column[i]-j>0)) (column[i-j]!=column[i]-j); %i abajo
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Solución planteada en clase:

constraint forall(i in 1..8, j in i+1..8)(
    column[i] !=  column[j] /\
    column[i] !=  column[j] + (j-i) /\
    column[i] !=  column[j] - (j-i)
);

solve satisfy;