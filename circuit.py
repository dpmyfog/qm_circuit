import numpy as np
import math


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
    return np.tensordot(np.tensordot(np.identity(2*(targetWire)), hadamardArray, 0), np.identity(2*(totalWires - targetWire - 1)), 0)       
		

def phase(theta, totalWires, targetWire):
    phaseArray = np.array([[1,0], [0, math.exp(1j * theta)]])
    return np.tensordot(np.tensordot(np.identity(2*(targetWire)), phaseArray, 0), np.identity(2*(totalWires - targetWire - 1)), 0)

def cnot(totalWires, controller, target):
    if totalWires - 1 == controller or totalWires - 1 == target: #special case to handle if one of them is the last one. if so, should not take last tensor product
        if controller == target - 1: #controller above target (assuming top wire is state |0>)
            cnotArray = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])
            return np.tensordot(np.identity(2*(controller)), cnotArray, 0)
        elif controller == target + 1: #controller below target (with top wire as |0>)
            cnotArray = np.array([[1, 0, 0 ,0],[0, 0, 0, 1],[0, 0, 1, 0],[0, 1, 0, 0]])
            return np.tensordot(np.identity(2*(target)), cnotArray, 0)

    elif controller == 0 or target == 0:#handles case if one is first. in this case, do not tensor prodcut with first empty matrix
        if controller == target - 1: #controller above target (assuming top wire is state |0>)
            cnotArray = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])
            return np.tensordot(cnotArray, np.identity(2*(totalWires - target - 1)), 0)
        elif controller == target + 1: #controller below target (with top wire as |0>)
            cnotArray = np.array([[1, 0, 0 ,0],[0, 0, 0, 1],[0, 0, 1, 0],[0, 1, 0, 0]])
            return np.tensordot(cnotArray, np.identity(2*(totalWires - controller - 1)), 0)

    else:
        if controller == target - 1: #controller above target (assuming top wire is state |0>)
            cnotArray = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])
            return np.tensordot(np.tensordot(np.identity(2*(controller)), cnotArray, 0), np.identity(2*(totalWires - target - 1)), 0)
        elif controller == target + 1: #controller below target (with top wire as |0>)
            cnotArray = np.array([[1, 0, 0 ,0],[0, 0, 0, 1],[0, 0, 1, 0],[0, 1, 0, 0]])
            return np.tensordot(np.tensordot(np.identity(2*(target)), cnotArray, 0), np.identity(2*(totalWires - controller - 1)), 0)

myInput = ReadInput("circuit/ex1")
print(myInput[0])
print(np.tensordot(np.identity(2), np.identity(2), 0))
print(cnot(3, 1, 2))
