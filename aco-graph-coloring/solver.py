import pymzn, json, time, operator
import math, networkx
from gurobipy import *
import os 
import argparse 

#!/usr/bin/python
# -*- coding: utf-8 -*-

INVALID_PATH_MSG = "Error: Invalid file path: "

METHOD_FLAGS={  'greedy':False,
                'mini':False,
                'local':False,
                'mip':False,
                }

input_data=""

TIME_FLAG=False

def validate_path(path):
    for p in path:
        if not os.path.exists(p): 
            print(INVALID_PATH_MSG + p)
            exit(-1)
        else:
            with open(p, 'r') as input_data_file:
                global input_data
                input_data = input_data_file.read()
                

def enable_methods(methods):
    for m in methods:
        if m in METHOD_FLAGS: METHOD_FLAGS[m] = True


def execute_methods():

    method = [k for k,v in METHOD_FLAGS.items() if v == True] [0]
    
    sol=0
    start=0

    if(TIME_FLAG): start = round(time.time() * 1000)

    if(method=='greedy'):
        sol=solve_it(input_data)

    elif (method=='mini'):
        sol=solve_it_minizinc(input_data)

    elif (method=='local'):
        sol=solve_it_local_search(input_data)
        
    elif(method=='mip'):
        sol=solve_it_MIP(input_data)
    
    print(sol)

    if(TIME_FLAG): print("Time: " + str(round(round(time.time() * 1000) - start)) + "ms")



   

def addcut(node_count,edges,model,color):
    G = networkx.Graph()
    G.add_edges_from(edges)
    for i in networkx.find_cliques(G):
        if len(i) == 3 or len(i) == 4:
            sumConstr = 0
            for pais in i:
                for j in range(0, node_count):
                    sumConstr += j * color[pais][j]
            if len(i) == 3:
                model.addConstr(sumConstr >= 3)
            else:
                model.addConstr(sumConstr >= 6)

def solve_it_MIP(input_data):
    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
        
    '''
    Variables de Decisión:
        Colores de los Países
    
    Restricciones:
        1 pais con 1 conexión tiene distintos colores
        
    Función Objetivo:
        Minimizar el número de colores
    '''

    model = Model()
    color = [[0 for i in range(0,node_count)]
            for j in range(0, node_count)]

    # Variables de decisión
    for i in range(0, len(color)):
        for j in range(0, node_count):
            color[i][j] = model.addVar(vtype=GRB.BINARY, name="color["+str(i)+"]"+"["+str(j)+"]")

    obj = model.addVar(vtype=GRB.CONTINUOUS, name="obj")

    # Restricciones

    for i in range(0, node_count):
        model.addConstr( quicksum(color[i]) == 1)

    for i in range(0, len(color)):
        sumConstr = 0
        for j in range(0, node_count):
            sumConstr += j * color[i][j]  
        model.addConstr(obj >= sumConstr)


    for i in range(0, len(edges)):
        for j in range(0, node_count):
            model.addConstr(color[edges[i][0]][j] + color[edges[i][1]][j] <= 1)
    

    addcut(node_count,edges,model,color)

    # Objetivo
    model.setObjective(obj,GRB.MINIMIZE)
    model.Params.OutputFlag = 0
    model.Params.timeLimit = 600
    model.Params.MIPGap = 0.02
    model.optimize()
    
    
    res = list()
    for i in range (0,node_count):
        for j in range(0,node_count):
            if color[i][j].getAttr("x") == 1 :
                res.append(j)

    # for m in model.getVars():
    #     if m.X > 0:
    #         print(m.varName, m.X)
        

    output_data = str(len(set(res)))  + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, res))

    return output_data
    

def getConstraints(edges):
    constraint = dict()
    for i in edges:
        if i[0] in constraint:
            constraint[i[0]] = constraint[i[0]] + 1
        else:
            constraint[i[0]] = 1
        if i[1] in constraint:
            constraint[i[1]] = constraint[i[1]] + 1
        else:
            constraint[i[1]] = 1

    return constraint
    
def auxRequirements(sol, edges):
    for i in edges:
        if sol[i[0]] == sol[i[1]]:
            return False
    return True

# Solución planteada con Búsqueda Local
def solve_it_local_search(input_data):
    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
        
    # build a trivial solution
    # every node has its own color
    greedy_solution = list(range(0, node_count))
    constraint = getConstraints(edges)

    notTakenList = list(range(0, node_count))
    takenList = dict()
    orderedList = list()
    
    while constraint:
        maxValue = max(constraint.items(), key=operator.itemgetter(1))[0]
        del constraint[maxValue]
        previous_val = greedy_solution[maxValue]
        
        momentaneumList = list()
        momentaneumList.extend(orderedList)
        momentaneumList.extend(notTakenList)
        orderedList.clear()

        for i in momentaneumList:
            greedy_solution[maxValue] = i
            if not auxRequirements(greedy_solution,edges):
                greedy_solution[maxValue] = previous_val
            else:
                if not i in takenList:
                    takenList[i] = 1
                    notTakenList.remove(i)
                else:
                    takenList[i] += 1

                a = sorted(takenList.items(), key=lambda x: x[1], reverse=True)
                for key in a:
                    orderedList.append(key[0])
                break

        if not auxRequirements(greedy_solution, edges):
            greedy_solution[maxValue] = previous_val
    
    # prepare the solution in the specified output format
    output_data = str(len(set(greedy_solution))) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, greedy_solution))

    return output_data

#Solución planteada con Minizinc (Programación con Restricciones).
def solve_it_minizinc(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    restriccionA = list()
    restriccionB = list()
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        restriccionA.append(int(parts[0]) + 1)
        restriccionB.append(int(parts[1]) + 1)

    countries = list(range(1,node_count + 1))
    s = pymzn.minizinc('solver.mzn', data = {'nSize':node_count , 'nRestrictions':edge_count, 'countries':countries, 'A':restriccionA, 'B':restriccionB})
    y = list(str(json.loads(str(s)[1:-1].replace("'",'"'))["coloresPaises"]).replace("[","").replace(",","").replace("]","").replace(' ', ''))
    
    for i in range(0, len(y)):
        y[i] = int(y[i]) - 1
    
    # prepare the solution in the specified output format
    output_data = str(len(set(y))) + ' ' + str(1) + '\n' + ' '.join(map(str, y))
    return output_data

def greedy(input_data):
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    graph = networkx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(edges)

    strategies = [networkx.coloring.strategy_largest_first,
                  networkx.coloring.strategy_random_sequential,
                  networkx.coloring.strategy_smallest_last,
                  networkx.coloring.strategy_independent_set,
                  networkx.coloring.strategy_connected_sequential_bfs,
                  networkx.coloring.strategy_connected_sequential_dfs,
                  networkx.coloring.strategy_connected_sequential,
                  networkx.coloring.strategy_saturation_largest_first]

    best_color_count, best_coloring = node_count, {i: i for i in range(node_count)}
    for strategy in strategies:
        curr_coloring = networkx.coloring.greedy_color(G=graph, strategy=strategy)
        curr_color_count = max(curr_coloring.values()) + 1
        if curr_color_count < best_color_count:
            best_color_count = curr_color_count
            best_coloring = curr_coloring
    obj, opt, solution= best_color_count, 0, [best_coloring[i] for i in range(node_count)]
    output_data = str(obj) + ' ' + str(opt) + '\n'
    output_data += ' '.join(map(str,solution))
    return output_data


#Solución planteada primitiva original.
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])  

    if node_count <= 200 :
        return solve_it_MIP(input_data)
    else:
        return greedy(input_data)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        parser = argparse.ArgumentParser() 
        parser.add_argument("-m", "--method", type = str, nargs = 1,
                            metavar = "method_name", default = None, required=True,
                            help = "Executes especified method.") 
        parser.add_argument("-p", "--path", type = str, nargs = 1,
                            metavar = "path_name", default = None, required=True,
                            help = "Takes input from path.") 

        parser.add_argument("-t", "--time", action="store_true",
                            help = "Shows execution time.") 

        args = parser.parse_args()
        
        validate_path(args.path)

        enable_methods(args.method)

        if(args.time): TIME_FLAG=True
        
        execute_methods()
