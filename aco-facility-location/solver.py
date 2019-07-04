#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from gurobipy import *
import math, time
import os 
import argparse 

INVALID_PATH_MSG = "Error: Invalid file path: "

METHOD_FLAGS={  'greedy':False,
                'mip':False,
                }

input_data=""

TIME_FLAG=False

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

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
      
    elif(method=='mip'):
        sol=gurobiFacility(input_data)
    
    print(sol)

    if(TIME_FLAG): print("Time: " + str(round(round(time.time() * 1000) - start)) + "ms")



def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def gurobiFacility(input_data):
    m = Model()
    m.Params.timeLimit = 600
    m.Params.MIPGap = 0.02
    m.Params.OutputFlag = 0
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    """
    Variables de Decisión:
    ci = cliente
    aj = almacén 
        Ci -> Aj
        Aj -> 1/0 [1 si se abre, 0 si no se abre]
    Restricciones:
        sum(Ci)<=1
        sum(demandas(Ci))<=Capacidad(Aj)
        Ci -> Aj si Aj es 1
    Función Objetivo:
        Minimizar C(t)
        C(t) = C(Aperturas) + C(Trayectos)
        C(Aperturas) = sum(Capertura(Aj)) si Aj es 1
        C(Trayectos) = sum(Cdistancia(Ci -> Aj))
    """
    
    # Variables de decisión -->
    cliente=[[0 for j in range(0, facility_count)]
             for i in range(0, customer_count)]

    almacen=[0 for i in range(0,facility_count)]

    
   
    for i in range(0,facility_count):
        almacen[i]= m.addVar(vtype=GRB.BINARY,name="almacen"+"["+ str(i) + "]" )

    for i in range(0,customer_count):
        for j in range(0, facility_count):
            cliente[i][j] = m.addVar(vtype=GRB.BINARY, name="cliente"+"["+str(i)+"]"+"["+str(j)+"]")

    m.update()
    # Restricciones -->
    for i in range(0,customer_count):
        m.addConstr(quicksum(cliente[i]) == 1)
        
    for i in range(0,facility_count):
        m.addConstr(quicksum(cliente[j][i] * customers[j].demand for j in range(0, customer_count)) <= facilities[i].capacity)
    
    for i in range(0,customer_count):
        for j in range(0, facility_count):
            m.addConstr(cliente[i][j] <= almacen[j])
    
    # Función Objetivo -->

    '''
    Función Objetivo:
        Minimizar C(t)
        C(t) = C(Aperturas) + C(Trayectos)
        C(Aperturas) = sum(Capertura(Aj)) si Aj es 1
        C(Trayectos) = sum(Cdistancia(Ci -> Aj))
    '''
 
    costeApertura = quicksum(almacen[i]*facilities[i].setup_cost for i in range(0, facility_count))
    
    costeTransporte=0

    for i in range(0,customer_count):
        for j in range(0, facility_count):
            costeTransporte += length(customers[i].location, facilities[j].location)*cliente[i][j]
        
    m.setObjective(costeApertura+costeTransporte,GRB.MINIMIZE)

    m.Params.OutputFlag = 0
    m.optimize()
    

    res=[0 for i in range(0,customer_count)]

    for i in range(0,customer_count):
        for j in range(0, facility_count):
            if cliente[i][j].getAttr("x") == 1 :
                res[i] = j


    output_data = '%.2f' % m.objVal + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, res))

    return output_data
    


def greedy(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    # Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
    # Customer = namedtuple("Customer", ['index', 'demand', 'location'])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    # build a trivial solution
    # pack the facilities one by one until all the customers are served
    solution = [-1]*len(customers)
    capacity_remaining = [f.capacity for f in facilities]

    facility_index = 0
    for customer in customers:
        if capacity_remaining[facility_index] >= customer.demand:
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand
        else:
            facility_index += 1
            assert capacity_remaining[facility_index] >= customer.demand
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand

    used = [0]*len(facilities)
    for facility_index in solution:
        used[facility_index] = 1

    # calculate the cost of the solution
    obj = sum([f.setup_cost*used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


def solve_it(input_data):
    return gurobiFacility(input_data)

import sys

if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        # Uso Profesional
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



