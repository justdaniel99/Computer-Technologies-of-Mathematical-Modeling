cimport cython
cimport numpy as np
import numpy as np
from cython.parallel import prange, parallel
from libc.math cimport sqrt, pow

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] acceleration(int body, double[:] masses, double[:] positions):
    
    cdef:
        int bodiesNumber = masses.size
        double gravitationConstant = 6.67408e-11
        double norm
        double[:] resultAcceleration = np.zeros(positions.size)
        double[:] temporaryArray = np.zeros(2)
        double distance
        int i, j, d


    for j in range(bodiesNumber):
        if body != j:
            
            norm = 0
            
            for d in range(2):
                distance = positions[j*2 + d] - positions[body*2 + d]
                norm += distance * distance
                temporaryArray[d] = gravitationConstant * masses[j] * distance
                
            norm = pow(norm, 1.5)
            
            for d in range(2):
                resultAcceleration[body*2 + d] += temporaryArray[d] / norm
    
    return resultAcceleration


@cython.boundscheck(False)
@cython.wraparound(False)
def cythonVerletAlgorithm(double[:] masses, double[:] initialPosition, double[:] initialVelocity, double deltaTime, int iterations):
    
    cdef:
        double[:] times = np.arange(iterations) * deltaTime
        double[:,:] position = np.empty((iterations, initialPosition.size))
        double[:,:] velocity = np.empty((iterations, initialPosition.size))
        double[:] currentAcceleration = np.zeros(initialPosition.size)
        double[:] nextAcceleration = np.zeros(initialPosition.size)
        int i, j, d, n
        int bodiesNumber = masses.size

    position[0] = initialPosition
    velocity[0] = initialVelocity
    
    for i in range(bodiesNumber):
        for d in range(2):
            currentAcceleration[i*2+d] = acceleration(i, masses, position[0])[d]
    
    for j in range(iterations - 1):
        
        for d in range(2):
            for n in range(bodiesNumber):
                position[j+1, n*2 + d] = position[j, n*2 + d] + velocity[j, n*2 + d] * deltaTime + 0.5 * currentAcceleration[n*2 + d] * deltaTime * deltaTime
        
        for i in range(bodiesNumber):
            for d in range(2):
                nextAcceleration[i*2+d] = acceleration(i, masses, position[j+1])[d]
        
        for d in range(2):
            for n in range(bodiesNumber):
                velocity[j+1, n*2 + d] = velocity[j, n*2 + d] + 0.5 * (currentAcceleration[n*2 + d] + nextAcceleration[n*2 + d]) * deltaTime
                
        currentAcceleration = nextAcceleration

    return position, velocity, times
