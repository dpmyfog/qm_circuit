import numpy as np
import math
import cmath

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


def hadamard(totalWires, targetWire):
    sq  = 1/math.sqrt(2)
    hadamardArray = np.array([[sq, sq],[sq, -1*sq]])
    hadamardArray = np.kron(np.identity(2*targetWire), hadamardArray)
    hadamardArray = np.kron(hadamardArray, np.identity(2*(totalWires - targetWire - 1)))
    return hadamardArray
		

def phase(totalWires, targetWire, phase):
    phaseArray = np.array([[1,0], [0, cmath.exp(1j * phase)]])
    
    hadamardArray = np.kron(np.identity(2*targetWire), phaseArray)
    hadamardArray = np.kron(phaseArray, np.identity(2*(totalWires - targetWire - 1)))

def cnot(totalWires, controller, target):
    if totalWires - 1 == controller or totalWires - 1 == target: #special case to handle if one of them is the last one. if so, should not take last tensor product
        if controller == target - 1: #controller above target (assuming top wire is state |0>)
            cnotArray = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])
            return np.kron(np.identity(2*(controller)), cnotArray, 0)
        elif controller == target + 1: #controller below target (with top wire as |0>)
            cnotArray = np.array([[1, 0, 0 ,0],[0, 0, 0, 1],[0, 0, 1, 0],[0, 1, 0, 0]])
            return np.kron(np.identity(2*(target)), cnotArray)

    elif controller == 0 or target == 0:#handles case if one is first. in this case, do not tensor prodcut with first empty matrix
        if controller == target - 1: #controller above target (assuming top wire is state |0>)
            cnotArray = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])
            return np.kron(cnotArray, np.identity(2*(totalWires - target - 1)))
        elif controller == target + 1: #controller below target (with top wire as |0>)
            cnotArray = np.array([[1, 0, 0 ,0],[0, 0, 0, 1],[0, 0, 1, 0],[0, 1, 0, 0]])
            return np.kron(cnotArray, np.identity(2*(totalWires - controller - 1)))

    else:
        if controller == target - 1: #controller above target (assuming top wire is state |0>)
            cnotArray = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])
            return np.kron(np.kron(np.identity(2*(controller)), cnotArray), np.identity(2*(totalWires - target - 1)))
        elif controller == target + 1: #controller below target (with top wire as |0>)
            cnotArray = np.array([[1, 0, 0 ,0],[0, 0, 0, 1],[0, 0, 1, 0],[0, 1, 0, 0]])
            return np.kron(np.kron(np.identity(2*(target)), cnotArray), np.identity(2*(totalWires - controller - 1)))


def process_circuit(filename):
    myInput = ReadInput(filename)
    totalWires = myInput[0]
    numGates = len(myInput[1])
    size = 2**totalWires
    gatesMatrix = np.identity(size)
    for i in xrange(numGates - 1, -1, -1): #process gates in reverse order
        if myInput[1][i][0] == 'H':
            gatesMatrix = gatesMatrix.dot(hadamard(float(totalWires), float(myInput[1][i][1])))
        elif myInput[1][i][0] == 'CNOT':
            gatesMatrix = gatesMatrix.dot(cnot(float(totalWires), float(myInput[1][i][1]), float(Input[1][i][2])))
        elif myInput[1][i][0] == 'P':
           gatesMatrix = gatesMatrix.dot(phase(float(totalWires), float(myInput[1][i][1]), float(myInput[1][i][2])))     
        elif myInput[1][i][0] == 'Measure':
            print('measure')
        return gatesMatrix
       
print(hadamard(3, 1))   
   

print(process_circuit('circuit/ex1'))
