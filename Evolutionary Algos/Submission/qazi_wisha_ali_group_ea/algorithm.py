from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from numpy import asarray
from scipy import misc
from scipy import stats
import scipy
import random

# Pass Array size as array tuple like array.shape()
# Returns a np.array 2d : [[22,334],[123,345],...]
def initialize_population(array_size,popluation_size):
    population = []

    x = np.random.randint(array_size[0],size=(popluation_size))
    y = np.random.randint(array_size[1],size=(popluation_size))
    
    
    for z in range(popluation_size):
        element = list((x[z],y[z]))
        population.append(element)
    return np.array(population)

# print(initialize_population(mainarray.shape,50).shape)

# This Function takes main image , babajee image and solution from population
def fitness_function(mainimage,babaimage,solution):
    x,y = babaimage.shape
    startx,starty= solution
    
    if startx+x < (mainimage.shape)[0] and starty+y < (mainimage.shape)[1]:
        result = mainimage[startx:startx+x,starty:starty+y]
        tau, p_value = stats.kendalltau(babaimage, result)
    
        return tau
    return -1.1111

# It returns list of population's Fitness
def get_fitness_of_all(population,mainimage,babaimage):
    fitness = []
    for individual in population:
        fit = float("{:.2f}".format(fitness_function(mainimage,babaimage,list(individual))))
        fitness.append(fit)
    return list(fitness)
# This function returns [[[x,y],correlation],.....]
def add_fitness_to_population(population,fitness):
    new_population = []
    for x in range(len(fitness)):
        temp = [population[x],fitness[x]]
        new_population.append(temp)
    return new_population

# This sorts population on basis of correlation
def sort_population(population):
    return sorted(population, key = lambda x: x[1],reverse=True)

# Takes 2 parents and returns 2 childs
def cross_over(parent_one , parent_two):
    # Get X , Y of each parent
    one_x , one_y = parent_one
    two_x , two_y = parent_two
    # Get binary rep and then combine in one number
    one_x = np.binary_repr(one_x,width=9)
    one_y = np.binary_repr(one_y,width=10)
    parent_one_binary = one_x+one_y

    two_x = np.binary_repr(two_x,width=9)
    two_y = np.binary_repr(two_y,width=10)
    parent_two_binary = two_x+two_y

    # Get Random Index for crossing over
    random_index = np.random.randint(0,18)

    # Parents crossed over
    child_one = parent_one_binary[:random_index]+parent_two_binary[random_index:]
    child_two = parent_two_binary[:random_index]+parent_one_binary[random_index:]

    # Separating their x and y
    child_one_x = child_one[:9]
    child_one_y = child_one[9:]
    child_two_x = child_two[:9]
    child_two_y = child_two[9:]
    child_one = list((int(child_one_x,2),int(child_one_y,2)))
    child_two = list((int(child_two_x,2),int(child_two_y,2)))
    return np.array(child_one), np.array(child_two)
# Takes individual and mutates it
def mutation(individual,fitness):
    x , y =  individual
    x = np.binary_repr(x,width=9)
    y = np.binary_repr(y,width=10)
    # Combined One Child
    one = x+y
    
    # NEW
    if fitness >= 0.70:
        random_index = np.random.randint(15,18)
        random_two = np.random.randint(7,9)
    elif fitness >= 0.50:
        random_index = np.random.randint(12,17)
        random_two = np.random.randint(5,9)
        
    elif fitness <= 0.35:    
        # random_index = np.random.randint(0,15)
        random_index = np.random.randint(9,17)
        random_two = np.random.randint(10,16)
    else:
        random_index = np.random.randint(1,10)
        random_two = np.random.randint(10,17)
        # random_index = np.random.randint(0,17)
        # random_two = np.random.randint(10,16)

    if one[random_index] == "0":
        one = one[:random_index] + "1" + one[random_index+1:]
    else:
        one = one[:random_index] + "0" + one[random_index+1:]
        # Got x and Y
    if fitness >= 0.70 :
        if one[random_index] == "0":
            one = one[:random_two] + "1" + one[random_two+1:]
        else:
            one = one[:random_two] + "0" + one[random_two+1:]


    # # 
    x = one[:9]
    y = one[9:]
    x = int(x,2)
    y = int(y,2)
    return np.array([x,y])
def createRandomSortedList(num, start = 1, end = 100): 
    arr = [] 
    tmp = random.randint(start, end) 
      
    for x in range(num): 
          
        while tmp in arr: 
            tmp = random.randint(start, end) 
              
        arr.append(tmp) 
          
    arr.sort() 
      
    return arr 
    
# Main Control begins
def main():
    # load image as pixel array
    number_of_generations = 0
    threshold = 0.9
    # best_fit_childs = [[[x,y],corre], ...... ]
    best_fit_childs = []
    mainFrame = Image.open('groupGray.jpg')
    baba = Image.open("boothiGray.jpg")

    mcols,mrows = mainFrame.size
    bcols,brows = baba.size
    print(f"Baba Jee Rows : {brows} baba jee cols : {bcols}")

    babaarray = asarray(baba)
    mainarray = asarray(mainFrame)

    population = initialize_population(mainarray.shape,50)
    
    fitness = get_fitness_of_all(population,mainarray,babaarray)
    
    pop_with_fitness = add_fitness_to_population(population,fitness)
    
    pop_with_fitness = sort_population(pop_with_fitness)
    # best_fit_childs.append(pop_with_fitness[0])
    maxi = -111111
    # Loop No of generations to a const number or threshold reached
    print(pop_with_fitness)
    print(pop_with_fitness[0])
    current_best = pop_with_fitness[0]
    gen_best = []
    print("Qazi Bes")
    print(pop_with_fitness[0])
    while True:
        childs = []
        # if pop_with_fitness[0][1] >= threshold:
        #     break
        if number_of_generations == 600:
            break
        if pop_with_fitness[0][1] >= threshold:
            print("Found")
            print(number_of_generations)
            break
        for x in range(0,50,2):
            parent_one = pop_with_fitness[x][0]
            parent_two = pop_with_fitness[x+1][0]
            child_one , child_two = cross_over(parent_one,parent_two)
           
            child_one = mutation(child_one,fitness_function(mainarray,babaarray,list(child_one)))
           
            child_two = mutation(child_two,fitness_function(mainarray,babaarray,list(child_two)))
            # child_one = mutation(child_one)
             
            child_one_fit = fitness_function(mainarray,babaarray,list(child_one))
            child_two_fit = fitness_function(mainarray,babaarray,list(child_two))
            childs.append([child_one,child_one_fit])
            childs.append([child_two,child_two_fit])
            
        # pop_with_fitness.extend(childs)
        childs = sort_population(childs)
        best_gen = childs[0]
        gen_best.append(childs[0][1])
        pop_with_fitness = pop_with_fitness[:4]
        pop_with_fitness.extend(childs[:46])
        pop_with_fitness = sort_population(pop_with_fitness)
        # pop_with_fitness = pop_with_fitness[:100]
        print("Best for next")
        print(pop_with_fitness[0])
        print("Best Child Gen from childs")
        print(best_gen)
        number_of_generations +=1
        best_fit_childs.append(pop_with_fitness[0])
        

    best_fit_childs = sort_population(best_fit_childs)
   
    print("My Generations")
    plt.figure()
    geens = [x for x in range(1,number_of_generations+1)]
    plt.plot(geens,gen_best,'green')
    plt.show()
    x , y = best_fit_childs[0][0]
    plt.imshow(mainFrame,cmap="gray")

    plt.scatter(y, x, s=50, c='red', marker='o')
    plt.scatter(639, 105, s=50, c='green', marker='o')
    plt.show()
    # Population sorted on fitness
    # max match 105 , 639 corr = 0.9923976390544154
    fit = fitness_function(mainarray,babaarray,[105,639])
    print(fit)
    print(fitness_function(mainarray,babaarray,[105, 635]))
    
    # plt.imshow(baba,cmap="gray")
    # plt.show()


# Main Called 
main()



