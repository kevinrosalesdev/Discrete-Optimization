**[Algoritmo de Graph Coloring implementado en Minizinc apoyado en Python usando Pymzn]**

**Este algoritmo incluye una mejora por Búsqueda Local**

Este algoritmo incluye:

- Algoritmo para definir el número de colores mínimos para colorear un grafo dado unas restricciones.
- Lista con los colores usados para cada zona.
- Mejora con Búsqueda Local en Python.
- Mejora con MIP.

Para ejecutar el algoritmo:
```
Greedy => 
python solver.py -m greedy -p [rutaDeArchivo]
Greedy Minizinc => 
python solver.py -m mini -p [rutaDeArchivo]
Local Search => 
python solver.py -m local -p [rutaDeArchivo]
MIP =>
python solver.py -m mip -p [rutaDeArchivo]
```

**Autores:**

- Doramas Báez Bernal
- Kevin Rosales Santana
- Marcos Jesus Santana Pérez

