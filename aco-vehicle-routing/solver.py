#!/usr/bin/python
# -*- coding: utf-8 -*-

import math, networkx, time
from collections import namedtuple
from gurobipy import *
import os 
import argparse 


INVALID_PATH_MSG = "Error: Invalid file path: "

METHOD_FLAGS={  'greedy':False,
                'mip':False,
                }

input_data=""

TIME_FLAG=False

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

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
        sol=gurobi_solution(input_data)

    print(sol)

    if(TIME_FLAG): print("Time: " + str(round(round(time.time() * 1000) - start)) + "ms")

# Método ideado para el formato correcto de printeo.
def subtour(trace, left, right):
    subtour = "0"
    if not 0 in left:
        return False
    pos = left.index(0)
    ini = right[pos]
    subtour += " " + str(ini)
    left[pos] = -1 
    right[pos] = -1 
    while ini != 0:
        for i in range(0, len(left)):
            if left[i] == ini:
                ini = right[i]
                subtour += " " + str(ini)
                left[i] = -1
                right[i] = -1
        if ini == 0:
            break
        for i in range(0, len(right)):
            if right[i] == ini:
                ini = left[i]
                subtour += " " + str(ini)
                left[i] = -1
                right[i] = -1
    return subtour

def gurobi_solution(input_data):
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    customers = []
    
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

    clientes = customers[1:]

    distancia_clientes = {}

    for i in customers:
        for j in customers:
            if i.index != j.index:
                distancia_clientes[i.index, j.index] = length(i,j)

    m = Model()
    m.Params.timeLimit = 600
    m.Params.MIPGap = 0.02
    # Variables de Decisión => Recorridos Usados y Capacidad de un vehículo al tratar con un cliente. 
    recorrido = {}
    for i in customers:
        for j in customers:
            if i.index != j.index:
                recorrido[i.index,j.index] = m.addVar(vtype=GRB.BINARY, name="recorrido[" + str(i.index) + "][" + str(j.index) + "]")
    
    capacidad_vehiculo = {}
    for i in customers:
        if i.index != 0:
            capacidad_vehiculo[i.index] = m.addVar(lb=i.demand, ub=vehicle_capacity, name="capacidad_vehiculo[" + str(i.index) + "]")

    # Restricciones =>
    # Solo puede haber una traza[i, CualquierColumna] y una traza [CualquierFila, j] (Por ejemplo = [1, 4] y [4, 1])

    for i in clientes:
        m.addConstr(quicksum(recorrido[i.index, j.index] for j in customers if i.index != j.index) == 1)

    for j in clientes:
        m.addConstr(quicksum(recorrido[i.index, j.index] for i in customers if i.index != j.index) == 1)

    # En caso de tratarse de una traza que parte del almacén, la capacidad del vehículo cuando trata con ese cliente debe ser la
    # demanda de ese cliente. En caso de no partir del almacén, se le da como margen a la capacidad actual del vehículo la capacidad
    # del vehículo genérica.
    for i in clientes:
        m.addConstr(capacidad_vehiculo[i.index] <= vehicle_capacity + (i.demand - vehicle_capacity)*recorrido[0,i.index])
    
    # En caso de tratarse de una traza real, la capacidad del vehículo cuando trató con el cliente anterior menos la capacidad del
    # vehículo con el nuevo cliente debe resultar en el valor negativo de la demanda del nuevo cliente. En caso de no tratarse de una
    # traza que se lleve a cabo, la diferencia deberá resultar en algo inferior a la diferencia entre la capacidad de un vehículo y
    # la demanda del nuevo cliente.
    for i in clientes:
        for j in clientes:
            if i != j:
                m.addConstr(capacidad_vehiculo[i.index] - capacidad_vehiculo[j.index] + vehicle_capacity*recorrido[i.index,j.index] <= vehicle_capacity - j.demand)
        
    # No pueden salir del almacén más coches de los marcados por vehicle_count
    m.addConstr(quicksum(recorrido[0, i.index] for i in clientes) <= vehicle_count)
    
    #Función Objetivo => Minimizar suma de las distancias de las trazas escogidas
    obj = quicksum(recorrido[i.index,j.index]*distancia_clientes[i.index, j.index] for i in customers for j in customers if i.index != j.index)

    m.setObjective(obj, GRB.MINIMIZE)
    m.Params.OutputFlag = 0
    m.optimize()
    
    left = list()
    right = list()
    for i in range(0, customer_count):
        for j in range(0, customer_count):
            if i != j and recorrido[i,j].getAttr("x") == 1 :
                left.append(i)
                right.append(j)
    
    res = list()
    for i in range(0, customer_count):
        for j in range(0, customer_count):
            if i != j and recorrido[i,j].getAttr("x") == 1 :
                route = subtour(recorrido, left, right)
                if route:
                    res.append(route)
    
    outputData = '%.2f' % m.objVal + ' ' + str(1) + '\n'
    for i in res:
        outputData += i + "\n"
    
    for i in range(0, vehicle_count-len(res)):
        outputData += "0  0\n"
        
    return outputData
    

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def greedy(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

    #the depot is always the first customer in the input
    depot = customers[0] 


    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = []
    
    remaining_customers = set(customers)
    remaining_customers.remove(depot)
    
    for v in range(0, vehicle_count):
        # print "Start Vehicle: ",v
        vehicle_tours.append([])
        capacity_remaining = vehicle_capacity
        while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
            used = set()
            order = sorted(remaining_customers, key=lambda customer: -customer.demand)
            for customer in order:
                if capacity_remaining >= customer.demand:
                    capacity_remaining -= customer.demand
                    vehicle_tours[v].append(customer)
                    # print '   add', ci, capacity_remaining
                    used.add(customer)
            remaining_customers -= used

    # checks that the number of customers served is correct
    assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1

    # calculate the cost of the solution; for each vehicle the length of the route
    obj = 0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 0:
            obj += length(depot,vehicle_tour[0])
            for i in range(0, len(vehicle_tour)-1):
                obj += length(vehicle_tour[i],vehicle_tour[i+1])
            obj += length(vehicle_tour[-1],depot)

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += str(depot.index) + ' ' + ' '.join([str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'

    return outputData

def solve_it(input_data):
    return gurobi_solution(input_data)

import sys


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



