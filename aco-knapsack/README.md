**[Algoritmo de KnapSack implementado en Python]** 

Este algoritmo incluye: 

```
- solve_it: Algoritmo Greedy por defecto.
- solve_it_optional_greedy: Algoritmo Greedy Mejorado.
- solve_it_memoization() & memoization(): Algoritmo de 'Memoization'.
- solve_it_tabulation(): Algoritmo de 'Tabulation'.
- solve_it_recursive_branchbound(): Algoritmo Recursivo en Profundidad de B&B (Con estimación).
- iterative_BranchBound(): Algoritmo Iterativo en Profundidad de B&B (Con estimación).
- coefficient_BranchBound(): Algoritmo Iterativo en Profundidad de B&B (Con estimación fraccionaria).
- best_first_BranchBound(): Algoritmo Iterativo de B&B utilizando la técnica de 'Mejor Primero' (Con estimación).
- solve_it_MIP(): Algoritmo que resuelve Knapsack usando MIP.
```

Para ejecutar solver.py, se requiere de la siguiente línea:
```
Greedy => 
python solver.py -m greedy -p [rutaDeArchivo]
Greedy Mejorado => 
python solver.py -m optgreedy -p [rutaDeArchivo]
Memoization => 
python solver.py -m memo -p [rutaDeArchivo]
Tabulation => 
python solver.py -m tabu -p [rutaDeArchivo]
B&B recursivo => 
python solver.py -m bbrec -p [rutaDeArchivo]
B&B iterativo => 
python solver.py -m bbiter -p [rutaDeArchivo]
B&B con mejor relajación => 
python solver.py -m bbrel -p [rutaDeArchivo]
B&B con mejor primero => 
python solver.py -m bbbf -p [rutaDeArchivo]
MIP => 
python solver.py -m mip -p [rutaDeArchivo]
```

**Autores:** 

- Doramas Báez Bernal 
- Kevin Rosales Santana
- Marcos Jesús Santana Pérez