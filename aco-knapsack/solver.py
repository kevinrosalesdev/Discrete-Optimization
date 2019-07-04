#!/usr/bin/python
# -*- coding: utf-8 -*-
from operator import attrgetter
from collections import namedtuple,deque
from collections import OrderedDict
from gurobipy import *
import time,os,argparse

INVALID_PATH_MSG = "Error: Invalid file path: "

METHOD_FLAGS={  'greedy':False,
                'optgreedy':False,
                'memo':False,
                'tabu':False,
                'bbrec':False,
                'bbiter':False,
                'bbrel':False,
                'bbbf':False,
                'mip':False,
                }

input_data=""

TIME_FLAG=False

# MAPAS USADOS EN ALGUNAS DE LAS IMPLEMENTACIONES
Item2 = namedtuple("Item", ['index', 'value', 'weight', 'coefficient'])
Item = namedtuple("Item", ['index', 'value', 'weight'])

# VALOR MÁXIMO USADO PARA COMPARACIONES
maxValue = 0


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
      
    elif(method=='optgreedy'):
        sol=solve_it_optional_greedy(input_data)

    elif(method=='memo'):
        sol=solve_it_memoization(input_data)

    elif(method=='tabu'):
        sol=solve_it_tabulation(input_data)

    elif(method=='bbrec'):
        sol=solve_it_recursive_branchbound(input_data)

    elif(method=='bbiter'):
        sol=iterative_BranchBound(input_data)

    elif(method=='bbrel'):
        sol=coefficient_BranchBound(input_data)

    elif(method=='bbbf'):
        sol=best_first_BranchBound(input_data)
    
    elif(method=='mip'):
        sol=solve_it_MIP(input_data)

    print(sol)
    
    if(TIME_FLAG): print("Time: " + str(round(round(time.time() * 1000) - start)) + "ms")



# NODO USADO EN LA IMPLEMENTACIÓN RECURSIVA
class node:
    def __init__(self,value,room,estimate,trace):
        self.value = value
        self.room = room
        self.estimate = estimate
        self.trace = trace.copy()

# NODO USADO EN EL RESTO DE IMPLEMENTACIONES
class iterative_node:
    def __init__(self,value,room,estimate,trace,index):
        self.value = value
        self.room = room
        self.estimate = estimate
        self.trace = trace.copy()
        self.index = index


def solve_it_MIP(input_data):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    room = int(firstLine[1])

    weights = list()
    values = list()

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        values.append(int(parts[0]))
        weights.append(int(parts[1]))
    
    m = Model("Knapsack")

    #Variables de Decisión
    taken = [0 for i in range(0, item_count)]
    for i in range(0, item_count):
        taken[i] = m.addVar(vtype=GRB.BINARY, name="Item["+str(i)+"]")

    #Restricciones
    m.addConstr(quicksum(taken[i]*weights[i] for i in range(0, item_count)) <= room)

    #Función Objetivo
    m.setObjective(quicksum(taken[i]*values[i] for i in range(0, item_count)), GRB.MAXIMIZE)

    m.setParam('OutputFlag', False)
    m.optimize()

    #for v in m.getVars():
    #    print(v.varName, v.X)
    
    #print("Objetivo => ", m.objVal)

    res = list()
    for i in range(0, item_count):
        res.append(int(taken[i].X))

    output_data = str(int(m.objVal)) + " 1\n" + ' '.join(map(str,res))
    return output_data



# MÉTODO QUE CALCULA EL ESTIMADO UTILIZANDO RELAJACIÓN FRACCIONARIA
def calculateNewEstimate(coefficient,room,values,weights,invalid):

    sorted_coefficient = OrderedDict(sorted(coefficient.items(), key=lambda x: x[1]))
    value = list(coefficient.values())
    estimate = 0
    for i in sorted_coefficient:

        if room - weights[value.index(sorted_coefficient[i])] >= 0 and invalid < value.index(sorted_coefficient[i]):
            room = room - weights[value.index(sorted_coefficient[i])]
            estimate += values[value.index(sorted_coefficient[i])]
            value[value.index(sorted_coefficient[i])] = -1

        elif room > 0 and invalid < value.index(sorted_coefficient[i]):
            aux = room / weights[value.index(sorted_coefficient[i])]
            estimate += values[value.index(sorted_coefficient[i])] * aux
            break

        elif room == 0:
            break

    return estimate

# MÉTODO QUE REALIZA EL BRANCH & BOUND UTILIZANDO LA TÉCNICA 'MEJOR PRIMERO'
def best_first_BranchBound(input_data):
    # parse the input
    global maxValue
    maxValue = 0
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    room = int(firstLine[1])

    weights = list()
    values = list()
    count = 0


    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        count = count + int(parts[0])
        values.append(int(parts[0]))
        weights.append(int(parts[1]))

    root = iterative_node(0,room,count,list(),-1)

    container = list()
    container.append(root)
    while container:
        root = container.pop()
        if root.estimate <= maxValue:
            break
        if root.index+1 == item_count and root.value > maxValue:
            bestNode = root
            maxValue = bestNode.value
            continue
        elif root.index+1 == item_count:
            continue
        else:
            if root.estimate - values[root.index+1] >= maxValue:
                newNodeRight = iterative_node(root.value, root.room, root.estimate-values[root.index+1],root.trace,root.index+1)
                newNodeRight.trace.append(0)
                container.append(newNodeRight)

            if root.room - weights[root.index+1] >= 0:
                newNodeLeft = iterative_node(values[root.index+1] + root.value, root.room-weights[root.index+1],root.estimate,root.trace,root.index+1)
                newNodeLeft.trace.append(1)
                container.append(newNodeLeft)
            
            container.sort(key=lambda node: node.estimate) # de menor a mayor
    output_data = str(maxValue) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, bestNode.trace))

    return output_data


# MÉTODO ITERATIVO QUE REALIZA BRANCH & BOUND EN PROFUNDIDAD CON ESTIMACIÓN FRACCIONARIA
def coefficient_BranchBound(input_data):
    # parse the input
    global maxValue
    maxValue = 0
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    room = int(firstLine[1])

    weights = list()
    values = list()
    coefficient = dict()


    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        values.append(int(parts[0]))
        weights.append(int(parts[1]))
        coefficient[i - 1] = int(parts[1])/int(parts[0])


    estimate = calculateNewEstimate(coefficient,room,values,weights,-1)
    root = iterative_node(0, room, estimate, list(), -1)
    
    container = deque()
    container.append(root)

    while container:

        root = container.pop()
        if root.index + 1 >= item_count and root.value >= maxValue:
            bestNode = root
            maxValue = bestNode.value
            continue
        elif root.index + 1 >= item_count:
            continue
        else:
            estimate = root.value + calculateNewEstimate(coefficient, root.room, values, weights,root.index+1)

            if estimate >= maxValue:
                newNodeRight = iterative_node(root.value, root.room, estimate, root.trace, root.index + 1)
                newNodeRight.trace.append(0)
                container.append(newNodeRight)

            if root.room - weights[root.index + 1] >= 0:
                newNodeLeft = iterative_node(values[root.index + 1] + root.value, root.room - weights[root.index + 1],
                                             root.estimate, root.trace, root.index + 1)
                newNodeLeft.trace.append(1)
                container.append(newNodeLeft)    

    output_data = str(maxValue) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, bestNode.trace))

    return output_data

# MÉTODO ITERATIVO QUE REALIZA BRANCH & BOUND EN PROFUNDIDAD
def iterative_BranchBound(input_data):
    # parse the input
    global maxValue
    maxValue = 0
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    room = int(firstLine[1])

    weights = list()
    values = list()
    count = 0


    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        count = count + int(parts[0])
        values.append(int(parts[0]))
        weights.append(int(parts[1]))

    root = iterative_node(0,room,count,list(),-1)

    container = deque()
    container.append(root)
    while container:
        root = container.pop()
        if root.index+1 >= item_count and root.value >= maxValue:
            bestNode = root
            maxValue = bestNode.value
            continue
        elif root.index+1 >= item_count:
            continue
        else:
            if root.estimate - values[root.index+1] >= maxValue:
                newNodeRight = iterative_node(root.value, root.room, root.estimate-values[root.index+1],root.trace,root.index+1)
                newNodeRight.trace.append(0)
                container.append(newNodeRight)

            if root.room - weights[root.index+1] >= 0:
                newNodeLeft = iterative_node(values[root.index+1] + root.value, root.room-weights[root.index+1],root.estimate,root.trace,root.index+1)
                newNodeLeft.trace.append(1)
                container.append(newNodeLeft)

    output_data = str(maxValue) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, bestNode.trace))

    return output_data

# MÉTODO QUE LLAMA A UN MÉTODO RECURSIVO PARA REALIZAR LA PILA EN B&B (NO RECOMENDADO)
def solve_it_recursive_branchbound(input_data):
    # parse the input
    global maxValue
    maxValue = 0
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    room = int(firstLine[1])

    weights = list()
    values = list()
    count = 0


    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        count = count + int(parts[0])
        values.append(int(parts[0]))
        weights.append(int(parts[1]))

    trace = list()
    root = node(0,room,count,trace)

    container = deque()
    container.append(root)
    container = addContainer(values,weights,root,item_count,0,container,trace)
    for i in container:
        if i.value == maxValue:
            maxNode = i

    output_data = str(maxValue) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, maxNode.trace))

    return output_data

# MÉTODO RECURSIVO QUE ITERA SOBRE LA PILA
def addContainer(values,weights,root,item_count,index,container,trace):

    global maxValue
    if index >= item_count:
        return -1
    else:
        if root.room - weights[index] >= 0 :
            trace.append(1)
            newNodeLeft = node(values[index] + root.value, root.room-weights[index],root.estimate,trace)
            container.appendleft(newNodeLeft)
            aux = addContainer(values, weights, newNodeLeft, item_count, index + 1, container,trace)
            trace.pop()
            if aux == -1 and newNodeLeft.value  > maxValue:
                maxValue = newNodeLeft.value
            elif aux == -1 and newNodeLeft.value  < maxValue:
                container.remove(root.nextValLeft)

        if root.estimate-values[index] >= maxValue:
            trace.append(0)
            newNodeRight = node(root.value, root.room, root.estimate - values[index], trace)
            container.remove(root)
            container.append(newNodeRight)
            aux = addContainer(values, weights, newNodeRight, item_count, index + 1, container, trace)
            trace.pop()

            if  aux == -1 and newNodeRight.value > maxValue:
                maxValue = newNodeRight.value
            elif aux == -1 and newNodeRight.value < maxValue:
                container.remove(root.nextValRight)

    return container

# MÉTODO QUE DEVUELVE LA TRAZA DE TABULATION
def getTabTrace(table, item_count, weights, capacity):
    tabTrace = list()
    i = item_count
    k = capacity
    while i > 0 and k > 0:
        if table[i][k] != table[i-1][k]:
            tabTrace.insert(0,1)
            i = i - 1
            k = k - weights[i]
        else:
            tabTrace.insert(0,0)
            i = i - 1
    for i in range(len(tabTrace),item_count):
        tabTrace.insert(0,0)
    return tabTrace

# Method that uses Tabulation algorithm to get result.
def solve_it_tabulation(input_data):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    weights = list()
    values = list()

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        values.append(int(parts[0]))
        weights.append(int(parts[1]))

    table = [[0 for x in range(capacity + 1)]
             for x in range(item_count + 1)]

    for i in range(1,item_count + 1):
        for w in range(0, capacity + 1):
            if weights[i-1] <= w:
                if values[i-1] + table[i-1][w - weights[i-1]] > table[i-1][w]:
                    table[i][w] = values[i-1] + table[i-1][w - weights[i-1]]
                else:
                    table[i][w] = table[i-1][w]
            else:
                table[i][w] = table [i-1][w]

    # prepare the solution in the specified output format

    tabTrace = getTabTrace(table, item_count, weights, capacity)
    output_data = str(table[item_count][capacity]) + ' ' + str(1) + '\n' +  ' '.join(map(str, tabTrace))
    return output_data

# ALGORITMO RECURSIVO DE MEMOIZATION
def Memoization(values, weights, capacity, item_count, table):
    if table.get(capacity*len(values) - item_count) is None:

        if (capacity == 0) or (item_count == 0):
            return 0

        if weights[item_count - 1] > capacity:
            return Memoization(values, weights, capacity, item_count - 1, table)
        else:
            case1 = values[item_count - 1] + Memoization(values, weights, capacity - weights[item_count - 1],item_count - 1, table)
            case2 = Memoization(values, weights, capacity, item_count - 1, table)
            table[capacity*len(values) - item_count] = max(case1,case2)

    return table.get(capacity*len(values) - item_count)

# Method that uses Memoization recursive algorithm to get result.
def solve_it_memoization(input_data):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    table = dict()
    weights = list()
    values = list()

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        values.append(int(parts[0]))
        weights.append(int(parts[1]))

    value = Memoization(values, weights, capacity, item_count, table)
    # prepare the solution in the specified output format

    output_data = str(value) + ' ' + str(1) + '\n'
    return output_data

#SOLUCIÓN GREEDY MEJORADA
def solve_it_optional_greedy(input_data):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []
    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        if int(parts[0]) == 0:
            continue
        # we use the coefficient to try to get the most valuable items and sort them
        coefficient = int(parts[1]) / int(parts[0])
        items.append(Item2(i - 1, int(parts[0]), int(parts[1]), coefficient))
    items = sorted(items, key=attrgetter('coefficient'))

    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0] * len(items)

    for item in items:

        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

# SOLUCIÓN GREEDY INICIAL
def solve_it(input_data):

    return solve_it_MIP(input_data)

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



