import pymzn, json, time
from collections import namedtuple
import os 
import argparse 
from gurobipy import *

INVALID_PATH_MSG = "Error: Invalid file path: "

METHOD_FLAGS={  'mip':False,
                'greedy':False,
                'mini':False,
                }

input_data=""

TIME_FLAG=False

Set = namedtuple("Set", ['index', 'cost', 'items'])


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
        sol=greedy(input_data)
      
    elif(method=='mini'):
        sol=setCovering(input_data)
    
    elif(method=='mip'):
        sol=solve_it_MIP(input_data)
    
    print(sol)

    if(TIME_FLAG): print("Time: " + str(round(round(time.time() * 1000) - start)) + "ms")


# Solución con MIP
def solve_it_MIP(input_data):

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    item_count = int(parts[0])
    set_count = int(parts[1])
    
    sets = []
    setsList = list()
    for i in range(1, set_count+1):
        parts = lines[i].split()
        sets.append(Set(i-1, float(parts[0]), map(int, parts[1:])))
        subList = list()
        for j in parts[1:]:
            subList.append(int(j))
        setsList.append(subList)

    m = Model()
    m.Params.OutputFlag = 0
    m.Params.timeLimit = 600
    m.Params.MIPGap = 0.02
    sol = [0 for i in range(0, set_count)]

    for i in range(0, set_count):
        sol[i] = m.addVar(vtype=GRB.BINARY, name='conjunto[' + str(i) + ']')

    for node in range(0, item_count):
        suma = 0
        for nset in range(0, set_count):
            ocurrencias = 0
            for set_item in setsList[nset]:
                if node == set_item:
                    ocurrencias += 1
            suma += sol[nset]*ocurrencias
        m.addConstr(suma >= 1)

    # calculate the cost of the solution
    m.setObjective(quicksum([s.cost*sol[s.index] for s in sets]),GRB.MINIMIZE)
    m.optimize()

    res = list()
    for i in range(0, set_count):
        res.append(int(sol[i].X))

    # prepare the solution in the specified output format
    output_data = str(m.objVal) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, res))

    return output_data

# Solución con Programación con Restricciones (Usando MiniZinc)
def setCovering(input_data):
     # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    item_count = int(parts[0])
    set_count = int(parts[1])
    
    cost = list()
    conjunto = list()
    for i in range(1, set_count+1):
        parts = lines[i].split()
        cost.append(int(parts[0]))
        conjunto.append(set([int(i) for i in parts[1:]]))

    s = pymzn.minizinc('solver.mzn', data = {'ITEM_COUNT':item_count, 'NSET_COUNT':set_count, 'sets':conjunto, 'cost':cost})
    y = json.loads('{"' + str(s)[str(s).index("selected"):-1].replace("'",'"'))
    
    # prepare the solution in the specified output format
    costeTotal = 0.0

    for i in range(0,len(cost)):
        costeTotal += y["selected_ones"][i]*cost[i]

    output_data = str(costeTotal) + ' ' + str(1) + '\n' + str(y["selected_ones"]).replace("[","").replace(",","").replace("]","") 
    return output_data

# Solución Primitiva Original
def greedy(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    item_count = int(parts[0])
    set_count = int(parts[1])
    
    sets = []
    for i in range(1, set_count+1):
        parts = lines[i].split()
        sets.append(Set(i-1, float(parts[0]), map(int, parts[1:])))

    # build a trivial solution
    # pick add sets one-by-one until all the items are covered
    solution = [0]*set_count
    coverted = set()
    
    for s in sets:
        solution[s.index] = 1
        coverted |= set(s.items)
        if len(coverted) >= item_count:
            break
        
    # calculate the cost of the solution
    obj = sum([s.cost*solution[s.index] for s in sets])

    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])  

    if node_count <= 75 :
        return setCovering(input_data)
    else:
        return solve_it_MIP(input_data)


import sys

if __name__ == '__main__':
    if len(sys.argv) == 2:
        # Uso Profesional
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        # Uso Academico
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
