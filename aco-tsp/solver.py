#!/usr/bin/python
# -*- coding: utf-8 -*-

import math,time,random,sys,numpy
from collections import namedtuple
import os, argparse 
import networkx as nx
import matplotlib.pyplot as plt

#import matplotlib; #buscar por mx.draw

INVALID_PATH_MSG = "Error: Invalid file path: "

METHOD_FLAGS={  'greedy':False,
                '2approx':False,
                'christo':False,
                '2opt_christo':False,
                '2opt_2approx':False,
                '3opt_christo':False,
                '3opt_2approx':False,
                'simua':False,
                'simutabu':False,
                'genetic':False,
                'ant':False,
                }

input_data=""

TIME_FLAG=False


Point = namedtuple("Point", ['x', 'y'])
points = []
res = []
preorder = []
tabuCocido = []
simulated = []
opt2_result = []


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
      
    elif(method=='2approx'):
        sol=approximation2(createGraph(input_data), False)
    
    elif(method=='christo'):
        sol=christofides(createGraph(input_data), False)

    elif(method=='2opt_christo'):
        christofides(createGraph(input_data), False)
        sol=opt_2(res.copy())
    
    elif(method=='2opt_2approx'):
        approximation2(createGraph(input_data), False)
        sol=opt_2(preorder.copy())

    elif(method=='3opt_christo'):
        christofides(createGraph(input_data), False)
        sol=opt_3(res.copy())

    elif(method=='3opt_2approx'):
        approximation2(createGraph(input_data), False)
        sol=opt_3(preorder.copy())

    elif(method=='simua'):
        christofides(createGraph(input_data), False)
        sol=simulated_annealing(res.copy())

    elif(method=='simutabu'):
        christofides(createGraph(input_data), False)
        sol=tabu_annealing(res.copy())

    elif(method=='genetic'):
        christofides(createGraph(input_data), False)
        tabu_annealing(res.copy())
        simulated_annealing(res.copy())
        sol=genetic_cycle(tabuCocido,simulated)

    elif(method=='ant'):
        christofides(createGraph(input_data), False)
        sol=ant_algorithm()
    
    
    print(sol)

    if(TIME_FLAG): print("Time: " + str(round(round(time.time() * 1000) - start)) + "ms")



def ant_algorithm():
    global points

    initial_pt = 5
    P = 0.9
    maxFlag = 3
    ants = round(len(points))

    min_value = math.inf
    min_trace = list()
    flag_flag = 0
    T = numpy.zeros((len(points),len(points)))
    flag = 0
    iter_min_value = math.inf
    pheromon_trace = [[initial_pt for j in range(0, len(points))] for i in range(0, len(points))]
    inverse_dist = [[0 for j in range(0, len(points))] for i in range(0, len(points))]
    for i in range(0,len(points)):
        for j in range(0,len(points)):
            if (i != j):
                inverse_dist[i][j] = (1/length(points[i],points[j]))
    
    randi_list = list()
    for _ in range(0,ants):
        randi_list.append(random.randrange(0,len(points)))
    while flag < maxFlag:
        for ant in randi_list:
            ant_solution = list()
            ant_solution.append(ant)
            for ant_row in ant_solution:
                prob = random.random()
                correct_indexes = list()
                pheromon_row = list()

                for i in range(0, len(points)):
                    if i not in ant_solution:
                        correct_indexes.append(i)

                for ant_column in range(0, len(points)):
                    if not ant_column in correct_indexes:
                        pheromon_row.append(-1)
                    else:
                        numerador= inverse_dist[ant_row][ant_column]*pheromon_trace[ant_row][ant_column]
                        denominador = sum([(numpy.multiply(inverse_dist[ant_row][j], pheromon_trace[ant_row][j])) for j in correct_indexes])
                        if denominador == 0:
                            pheromon_row.append(-1)
                            continue
                        pheromon_row.append(numerador/denominador)
                        pheromon_weight = 0
                        for i in range(0,len(pheromon_row)):
                            if pheromon_row[i] != -1:
                                pheromon_weight += pheromon_row[i]
                                if prob < pheromon_weight:
                                    ant_solution.append(i)
                                    flag_flag = 1
                                    break

                    if flag_flag == 1:
                        flag_flag = 0
                        break
            contador = 0
            for i in range(0, len(ant_solution)-1):
                distancia = length(points[ant_solution[i]], points[ant_solution[i+1]])
                contador += distancia

            contador+=length(points[ant_solution[-1]], points[ant_solution[0]])
            if contador < min_value:
                min_value = contador
                min_trace = ant_solution

            inv_contador = 1/contador
            for i in range(0, len(points)-1):
                T[ant_solution[i]][ant_solution[i+1]] += inv_contador
            T[ant_solution[-1]][ant_solution[0]] += inv_contador

        if min_value == iter_min_value:
            flag += 1

        iter_min_value = min_value

        pheromon_trace = numpy.multiply((1-P),pheromon_trace) + T
        
    solution = normalize(min_trace)    
    output_data = '%.2f' % min_value + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))
    #output_data += ' ' + str(0)

    return output_data
    
def tabu_annealing(x):
    T = 10
    N = 420
    flag = 0
    alpha = 0.3
    tabu_list = list()
    tabu_max = len(x)/4
    tabu_counter = 0
    tabu_flag = 0
    
    while flag != 3:
        for i in range(0,N):
            a = random.randrange(1, len(x)-2)
            b = random.randrange(a+2, len(x))
            Cy = distance(x,a-1,b-1) + distance(x,a,b) 
            Cx = distance(x,a-1,a) + distance(x,b-1,b) 
            delta = Cy-Cx

            if tabu_counter == tabu_max:
                tabu_counter = 0
                del tabu_list[0]

            for i in tabu_list:
                if x[a] == i[0] and x[b] == i[1]:
                    tabu_flag = 1
                    break

            if tabu_flag == 1:
                tabu_flag = 0
                continue

            if delta < 0:
                tabu_list.append([x[a],x[b-1]])
                x[a:b] = reversed(x[a:b])
                flag = 0
                tabu_counter += 1
            elif delta >= 0 and random.random() < math.exp(-delta/T):
                tabu_list.append([x[a],x[b-1]])
                x[a:b] = reversed(x[a:b])
                flag += 1
                tabu_counter += 1
                break
            
        T = alpha*T
        if (T == 0):
            break
    
    contador = 0
    for i in range(0, len(x)-1):
        distancia = length(points[x[i]], points[x[i+1]])
        contador += distancia
        
    contador+=length(points[x[-1]], points[x[0]])
    
    output_data = '%.2f' % contador + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, x))
    #output_data += ' ' + str(x[0])

    global tabuCocido
    tabuCocido = x.copy()
    return output_data

def simulated_annealing(x):

    T = 10
    N = 1000
    flag = 0
    alpha = 0.95
    while flag != 1000000:
        for i in range(0,N):
            a = random.randrange(1, len(x)-2)
            b = random.randrange(a+2, len(x))
            Cy = distance(x,a-1,b-1) + distance(x,a,b) 
            Cx = distance(x,a-1,a) + distance(x,b-1,b) 
            delta = Cy-Cx
            if delta < 0:
                x[a:b] = reversed(x[a:b])
                flag = 0

            elif delta >= 0 and random.random() < math.exp(-delta/T):
                x[a:b] = reversed(x[a:b])
                flag += 1
                
                break   #Si empeoramos, enfriamos
        T = alpha*T
        if (T < 0.00001):
            break
    

    contador = 0
    for i in range(0, len(x)-1):
        distancia = length(points[x[i]], points[x[i+1]])
        contador += distancia

    contador+=length(points[x[-1]], points[x[0]])
    
    output_data = '%.2f' % contador + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, x))
    #output_data += ' ' + str(x[0])

    global simulated
    simulated = x.copy()
    return output_data

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def createGraph(input_data):
    lines = input_data.split('\n')

    nodeCount = int(lines[0])
    g = nx.Graph() #creamos grafo
    global points
    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    for i in range(0, nodeCount):
        for j in range(i+1, nodeCount):
            g.add_edge(i,j, weight = length(points[i], points[j]))
    
    #print("n nodes: ",g.number_of_nodes(), "n edges: ", g.number_of_edges())
    return g

def approximation2(g, printDraw = False):
    
    H= nx.empty_graph()

    tree = nx.minimum_spanning_tree(g)
    # You might want to use the function "nx.minimum_spanning_tree(g)"
    # which returns a Minimum Spanning Tree of the graph 
    global preorder
    preorder = list(nx.dfs_preorder_nodes(tree))
    # You also might want to use the command "list(nx.dfs_preorder_nodes(graph, 0))"
    # which gives a list of vertices of the given graph in depth-first preorder.
    
    contador=0
    global points

    for i in range(0, len(preorder)-1):
        distancia = length(points[preorder[i]], points[preorder[i+1]])
        contador += distancia
        H.add_edge(preorder[i], preorder[i+1], weight = distancia)
        
    contador+=length(points[preorder[len(preorder)-1]], points[preorder[0]])
    H.add_edge(preorder[len(preorder)-1], preorder[0], weight = distancia)

    output_data = '%.2f' % contador + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, preorder))
    #output_data += ' ' + str(preorder[0])
    
    if printDraw:
        pos = nx.spring_layout(H)
        nx.draw_networkx_nodes(H, pos, node_size=175)
        nx.draw_networkx_nodes(H, pos, nodelist=[preorder[0], preorder[len(preorder)-1]], node_size=175, node_color='y')
        nx.draw_networkx_edges(H, pos, width=1)
        nx.draw_networkx_labels(H, pos, font_size=9, font_family='sans-serif')
        plt.show()
    return output_data

def christofides(g, printDraw = False):
    # n is the number of vertices.
    # H -> Multigrafo, Hres -> Grafo representativo, Hodd -> Grafo de conexiones impares
    H= nx.MultiGraph()
    Hres = nx.empty_graph()
    Hodd = nx.empty_graph()

    tree = nx.minimum_spanning_tree(g)
    # You might want to use the function "nx.minimum_spanning_tree(g)"
    # which returns a Minimum Spanning Tree of the graph g
    global points
    conexiones = tree.degree()
    impares = list()

    for i in range(0, len(conexiones)):
        if (conexiones[i] % 2 != 0):    # impares
            impares.append(i)

    # nx.max_weight_matching(W, maxcardinality = True)
    # which computes a maximum-weighted matching of W.

    for i in range(0, len(impares)):
        for j in range(i+1, len(impares)):
            Hodd.add_edge(impares[i],impares[j], weight = -length(points[impares[i]], points[impares[j]]))

    M = nx.max_weight_matching(Hodd, maxcardinality = True) #Emparejamiento optimo

    #impares
    for i in M:
        H.add_edge(i[0], i[1], weight =length(points[i[0]], points[i[1]]))

    #Básico
    for i in tree.edges(data='weight'):
        H.add_edge(i[0], i[1], weight = i[2])

    eulerian_circuit = []
    for i in nx.eulerian_circuit(H, source=0):
        eulerian_circuit.append(i[0])
    global res

    [res.append(item) for item in eulerian_circuit if item not in res]
    contador=0

    for i in range(0, len(res)-1):
        distancia = length(points[res[i]], points[res[i+1]])
        contador += distancia
        Hres.add_edge(res[i], res[i+1], weight = distancia)
        
    contador+=length(points[res[-1]], points[res[0]])
    Hres.add_edge(res[len(res)-1], res[0], weight = distancia)

    output_data = '%.2f' % contador + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, res))
    #output_data += ' ' + str(res[0])
    
    if printDraw:
        pos = nx.spring_layout(Hres)
        nx.draw_networkx_nodes(Hres, pos, node_size=175)
        nx.draw_networkx_nodes(Hres, pos, nodelist=[res[0], res[len(res)-1]], node_size=175, node_color='y')
        nx.draw_networkx_edges(Hres, pos, width=1)
        nx.draw_networkx_labels(Hres, pos, font_size=9, font_family='sans-serif')
        plt.show()
    return output_data

def distance(aux,a,b):
    return length(points[aux[a]], points[aux[b]])


def genetic_cycle(dad,mom):
    res = [-1 for _ in range(0,len(dad))]
    
    while -1 in res:
        iteration = random.randrange(0,2)
        starting_point=res.index(-1)
        valdad = dad[starting_point]
        valmom = mom[starting_point]

        if(iteration == 0): #daddy
            last_value = valdad
            res[starting_point] = valdad
            while last_value != valmom:
                pos_mom_in_dad = dad.index(valmom)
                res[pos_mom_in_dad] = valmom
                valmom = mom[pos_mom_in_dad]

        if(iteration == 1): #mom
            last_value = valmom
            res[starting_point] = valmom
            while last_value != valdad:
                pos_dad_in_mom = mom.index(valdad)
                res[pos_dad_in_mom] = valdad
                valdad = dad[pos_dad_in_mom]
    contador = 0
    for i in range(0, len(res)-1):
        distancia = length(points[res[i]], points[res[i+1]])
        contador += distancia
        
    contador+=length(points[res[-1]], points[res[0]])
    
    output_data = '%.2f' % contador + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, res))
    #output_data += ' ' + str(res[0])

    return output_data
            

def opt_2(res):
    for i in range(1,len(res)):
        for j in range (i+2,len(res)+1):

            a,b,c,d=i-1,i,j-1,j%len(res)

            if a<j:
                A=reversed(res[b:d])
            else:
                A= res[d:b] + list(reversed(res[b:]))

            if distance(res,a,c) + distance(res,b,d) < distance(res,a,b) + distance(res,c,d):
                res[b:d] = A
    global opt2_result            
    opt2_result = res.copy()
    contador = 0
    for i in range(0, len(res)-1):
        distancia = length(points[res[i]], points[res[i+1]])
        contador += distancia
        
    contador+=length(points[res[-1]], points[res[0]])
    
    output_data = '%.2f' % contador + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, res))
    #output_data += ' ' + str(res[0])

    return output_data

def smallest(l):
    return l.index(min(l))

def normalize(l):
    cero=smallest(l)
    return (l[cero:]+l[:cero])

def opt_3(res):
    for i in range(1,len(res)):
        for j in range(i+2, len(res)):
            for k in range(j+2, len(res)):
                a, b, c, d, e, f = i-1, i, j-1, j, k-1, (k)%len(res)
                dist=[0,0,0,0,0,0,0,0]
                
                dist[0] = distance(res,a,b) + distance(res,c,d) + distance(res,e,f) #ABC
                dist[1] = distance(res,f,b) + distance(res,c,d) + distance(res,e,a) #A'BC
                dist[2] = distance(res,a,b) + distance(res,c,e) + distance(res,d,f) #ABC'
                dist[3] = distance(res,f,b) + distance(res,c,e) + distance(res,d,a) #A'BC'
                dist[4] = distance(res,f,c) + distance(res,b,d) + distance(res,e,a) #A'B'C
                dist[5] = distance(res,a,c) + distance(res,b,d) + distance(res,e,f) #AB'C
                dist[6] = distance(res,a,c) + distance(res,b,e) + distance(res,d,f) #AB'C'
                dist[7] = distance(res,f,c) + distance(res,b,e) + distance(res,d,a) #A'B'C'

                min=smallest(dist)
                A = res[f:] + res[:b]
                B = res[b:d]
                C = res[d:f]

                if (min==1):
                    #A'BC
                    res= normalize( list(reversed(A)) + B + C )
                elif (min==2):
                    #ABC'
                    res= normalize( A + B + list(reversed(C)) )
                elif (min==3):
                    #A'BC'
                    res= normalize( list(reversed(A)) + B + list(reversed(C)) )
                elif (min==4):
                    #A'B'C
                    res= normalize( list(reversed(A)) + list(reversed(B)) + C )
                elif (min==5):
                    #AB'C
                    res= normalize( A + list(reversed(B)) + C )
                elif (min==6):
                    #AB'C'
                    res= normalize( A + list(reversed(B)) + list(reversed(C)) )
                elif (min==7):
                    #A'B'C'
                    res= normalize( list(reversed(A)) + list(reversed(B)) + list(reversed(C)) )
                        
    contador = 0
    for i in range(0, len(res)-1):
        distancia = length(points[res[i]], points[res[i+1]])
        contador += distancia
        
    contador+=length(points[res[-1]], points[res[0]])
    
    output_data = '%.2f' % contador + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, res))
    #output_data += ' ' + str(res[0])

    return output_data

def greedy(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])
    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    solution = range(0, nodeCount)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


def solve_it(input_data):
    lines = input_data.split('\n')

    nodeCount = int(lines[0])
    global res, opt2_result, preorder
    if nodeCount <= 400:
        christofides(createGraph(input_data))
        return opt_3(res)
    elif nodeCount > 400 and nodeCount < 2105:
        christofides(createGraph(input_data))
        opt_2(res)
        return simulated_annealing(opt2_result)
    elif nodeCount >= 2105 and nodeCount < 5000: 
        approximation2(createGraph(input_data))
        opt_2(preorder)
        return simulated_annealing(opt2_result)
    else:
        global points
        points = []
        for i in range(1, nodeCount+1):
            line = lines[i]
            parts = line.split()
            points.append(Point(float(parts[0]), float(parts[1])))
        if nodeCount >= 5000 and nodeCount < 25000:
            opt_2(list(range(0,nodeCount)))
            return simulated_annealing(opt2_result)
        else:
            return simulated_annealing(list(range(0,nodeCount)))


import sys

if __name__ == '__main__':
    if len(sys.argv) == 2:
        # Uso Profesional
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    elif len(sys.argv) > 1: 
        # Uso academico
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
    else: 
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')




    
    
