import numpy as np
import math
import cmath
from fractions import gcd
import random
import time
import matplotlib.pyplot as plt
np.set_printoptions(linewidth = 300)

def ReadInput(fileName):
    myInput_lines=open(fileName).readlines()
    myInput=[]
    numberOfWires=int(myInput_lines[0])
    for line in myInput_lines[1:]:
        myInput.append(line.split())
    return (numberOfWires,myInput) #returns tuple

'''
#example usage
myInput=ReadInput("myGateDescription")
firstGate=myInput[0][0]
secondGate=myInput[1][0]
firstWire=myInput[0][1]
'''

'''
def hadamard(totalWires, targetWire):
    sq  = 1/math.sqrt(2)
    hadamardArray = np.array([[sq, sq],[sq, -1*sq]])
    if targetWire == 0:
        #dot with Isize-1Xsize-1
        hadamardArray = np.kron(hadamardArray, np.identity(2**(totalWires - 1)))
    elif targetWire == totalWires - 1:
        #case of last wire
        hadamardArray = np.kron(np.identity(2**targetWire), hadamardArray)
    else: 
        hadamardArray = np.kron(np.identity(2**targetWire), hadamardArray)
        hadamardArray = np.kron(hadamardArray, np.identity(2**(totalWires - targetWire - 1)))
    return hadamardArray
		

def phase(totalWires, targetWire, phase):
    phaseArray = np.array([[1, 0],[0, cmath.exp(1j*phase)]])
    if targetWire == 0:
        #case of first wire
        phaseArray = np.kron(phaseArray, np.identity(2**(totalWires - 1)))
    elif targetWire == totalWires - 1:
        #case of last wire
        phaseArray = np.kron(np.identity(2**targetWire), phaseArray)
    else: 
        phaseArray = np.kron(np.identity(2**targetWire), phaseArray)
        phaseArray = np.kron(phaseArray, np.identity(2**(totalWires - targetWire - 1)))
    return phaseArray

def cnot(totalWires, controller, target):
    if (totalWires - 1 == controller) or (totalWires - 1 == target): #special case to handle if one of them is the last one. if so, should not take last tensor product
        if controller == target - 1: #controller above target (assuming top wire is state |0>)
            cnotArray = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])
            return np.kron(np.identity(2**(controller)), cnotArray, 0)
        elif controller == target + 1: #controller below target (with top wire as |0>)
            cnotArray = np.array([[1, 0, 0 ,0],[0, 0, 0, 1],[0, 0, 1, 0],[0, 1, 0, 0]])
            return np.kron(np.identity(2**(target)), cnotArray)

    elif (controller == 0) or (target == 0):#handles case if one is first. in this case, do not tensor prodcut with first empty matrix
        if controller == target - 1: #controller above target (assuming top wire is state |0>)
            cnotArray = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])
            return np.kron(cnotArray, np.identity(2**(totalWires - target - 1)))
        elif controller == target + 1: #controller below target (with top wire as |0>)
            cnotArray = np.array([[1, 0, 0 ,0],[0, 0, 0, 1],[0, 0, 1, 0],[0, 1, 0, 0]])
            return np.kron(cnotArray, np.identity(2**(totalWires - controller - 1)))

    else:
        if (controller == target - 1): #controller above target (assuming top wire is state |0>)
            cnotArray = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])
            return np.kron(np.kron(np.identity(2**(controller)), cnotArray), np.identity(2**(totalWires - target - 1)))
        elif (controller == target + 1): #controller below target (with top wire as |0>)
            cnotArray = np.array([[1, 0, 0 ,0],[0, 0, 0, 1],[0, 0, 1, 0],[0, 1, 0, 0]])
            return np.kron(np.kron(np.identity(2**(target)), cnotArray), np.identity(2**(totalWires - controller - 1)))

'''

def phase(totalWires, targetWire, phase):
    phaseArray = np.array([[1, 0],[0, cmath.exp(1j*phase)]])
    identity = np.identity(2)
    arrlist = []
    for i in range(totalWires):
        if i == targetWire:
            arrlist.append(phaseArray)
        else:
            arrlist.append(identity)
    #print(arrlist)
    return kron_list(arrlist)

def hadamard(totalWires, targetWire):
    sq  = 1/math.sqrt(2)
    hadamardArray = np.array([[sq, sq],[sq, -1*sq]])
    arrlist = []
    for i in range(totalWires):
        if i == targetWire:
            arrlist.append(hadamardArray)
        else:
            arrlist.append(np.identity(2))
    return kron_list(arrlist)

def cnot(totalWires, controller, target):
    arrlist = []
    if controller == target - 1: #controller above target (assuming top wire is state |0>)
            cnotArray = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])
            
    elif controller == target + 1: #controller below target (with top wire as |0>)
            cnotArray = np.array([[1, 0, 0 ,0],[0, 0, 0, 1],[0, 0, 1, 0],[0, 1, 0, 0]])
            
    for i in range(totalWires - 1): #minus one because two of the matrices have been condensed into one
        if i == min(controller, target):
            arrlist.append(cnotArray)
        else:
            arrlist.append(np.identity(2))
    
    return kron_list(arrlist)

'''
def pauli_x():
    return np.array([[1,0],[0,-1]]) 

def pauli_y():

def pauli_z():
'''
def kron_list(arrlist):
    outarr = arrlist[0]
    for i in range(1, len(arrlist)):
        outarr = np.kron(outarr, arrlist[i])
    return outarr

def process_circuit(filename):
    myInput = ReadInput(filename)
    totalWires = myInput[0]
    numGates = len(myInput[1])
    print(numGates)
    size = 2**totalWires
    gatesMatrix = np.identity(size)
    for i in xrange(numGates-1, -1, -1): #process gates in reverse order
        if myInput[1][i][0] == 'H':
            #print(totalWires)
            #print(myInput[1][i][1])
            #print(hadamard(float(totalWires), float(myInput[1][i][1])))
            gatesMatrix = gatesMatrix.dot(hadamard(int(totalWires), int(myInput[1][i][1])))
        elif myInput[1][i][0] == 'CNOT':
            gatesMatrix = gatesMatrix.dot(cnot(int(totalWires), int(myInput[1][i][1]), int(myInput[1][i][2])))
        elif myInput[1][i][0] == 'P':
            print('read phase')
            gatesMatrix = gatesMatrix.dot(phase(int(totalWires), int(myInput[1][i][1]), float(myInput[1][i][2])))     
        elif myInput[1][i][0] == 'Measure':
            print('measure')
    return gatesMatrix
       
def build_random(filename, length, numWires):
	#first create random circuit as a string, each element terminated with newline
	#can do phase, hadamard, nearest-neighbor CNOT
	description = str(numWires) + "\n"
	inverse = []
	for i in range(length):
		rnd = random.randint(1, 3)
		if rnd == 1: #hadamard
			addition = "H " + str(random.randint(0, numWires - 1))
                        description += addition + "\n"
                        inverse.append(addition) #hadamard is its own inverse
		elif rnd == 2: #phase
                        wire = random.randint(0, numWires-1)
                        phase = random.uniform(0, math.pi)
			addition = "P " + str(wire) + " " + str(phase)
                        invaddition = "P " + str(wire) + " " + str(-1*phase)
                        description += addition + "\n"
                        inverse.append(invaddition)
		elif rnd == 3: #CNot
			altWire = 0
			controlWire = random.randint(0, numWires - 1)
			if random.uniform(0, 1) > 0.5:
				altWire = 1
			else:
				altWire = -1
			addition = "CNOT " + str(controlWire) + " " + str(controlWire + altWire) 
                        description += addition + "\n"
                        inverse.append(addition) #cnot is its own inverse
                        
        #now inverse contains all of the information about the inverses of each matrix
        for i in range(len(inverse)):
            description += inverse[len(inverse) - i - 1] + '\n'
		
	with open(filename, 'w') as f:
		f.write(description)
        print description

def classical_shors(n):
    a = random.randint(0, int(math.sqrt(n)))
    #print(x)
    r = 0
    retval = (-1, -1)
    if gcd(a, n) != 1: # we found a nontrivial factor
        retval = (gcd(a, n), n/gcd(a , n))
    else: #need to complete rest of shor's
        r = 1
        while (a**r)%n != 1:
            r += 1
        if r%2 == 1: #if r is odd try again
            retval =  classical_shors(n)
        else: 
            first = gcd(a**(r/2) - 1, n)
            second = gcd(a**(r/2) + 1, n)
            if(first == 1 or second == 1):
                retval =  classical_shors(n)
            else:
                retval = (first, second)
    return retval


def time_function(myfnc, param = None):
    start = time.time()
    myfnc(param)
    end = time.time()
    return end-start


def plot_runtime(myfnc, domain):
    times = []
    for i in domain:
        print(time_function(myfnc, i))
        times.append(time_function(myfnc, i))
    plt.plot(domain, times)
    plt.show()


    
print(classical_shors(71**2))
plot_runtime(classical_shors, range(5, 70))
#print(ReadInput('circuit/ex1'))   
#print(hadamard(3,0))
#print(np.identity(8).dot(hadamard(3, 1)))


#build_random("circuit/random", 4, 3)
#print(process_circuit('circuit/random'))
