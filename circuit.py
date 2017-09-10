import numpy as np
import math


def ReadInput(fileName):
    myInput_lines=open(fileName).readlines()
    myInput=[]
    numberOfWires=int(myInput_lines[0])
    for line in myInput_lines[1:]:
        myInput.append(line.split())
    return (numberOfWires,myInput) 

'''
#example usage
myInput=ReadInput("myGateDescription")
firstGate=myInput[]0[0]
secondGate=myInput[1][0]
firstWire=myInput[0][1]
'''

def identity(): 
    return np.array([[1,0],[0,1]])

def hadamard():
    sq  = 1/math.sqrt(2)
    return np.array([[sq, sq],[sq, -1*sq]])

def phase(theta, ):
    return np.array([[1,0], [0, math.exp(1j * theta)])

                    def cnot(controller, target, totalWires):
    if controller == target - 1: #controller above target (assuming top wire is state |0>)
            return np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])
    elif controller == target - 1: #controller below target (with top wire as |0>)
            return np.array([[1, 0, 0 ,0],[0, 0, 0, 1],[0, 0, 1, 0],[0, 1, 0, 0]])
