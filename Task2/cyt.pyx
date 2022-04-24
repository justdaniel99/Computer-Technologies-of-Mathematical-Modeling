cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt, pow

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[double, ndim=1] acceleration(int body, np.ndarray[double, ndim=1] masses, np.ndarray[double, ndim=1] positions):
    cdef:
        int bodiesNumber = masses.size
        double norm
        double gravitationConstant = 6.67408e-11
        np.ndarray resultAcceleration = np.zeros(positions.size)
        np.ndarray distance = np.zeros(2)
        int j

    for j in range(bodiesNumber):
        if body != j:
            distance = positions[j*2:(j+1)*2] - positions[body*2:(body+1)*2]
            norm = pow(distance.dot(distance), 1.5)
            resultAcceleration[j*2:(j+1)*2] += gravitationConstant * masses[j] * distance / norm
            
    return resultAcceleration


@cython.boundscheck(False)
@cython.wraparound(False)
def cythonVerletAlgorithm(np.ndarray[double, ndim=1] masses, np.ndarray[double, ndim=1] initialPosition, np.ndarray[double, ndim=1] initialVelocity, double deltaTime, int iterations):
    cdef:
        times = np.arange(iterations) * deltaTime
        int bodiesNumber = masses.size
        np.ndarray[double, ndim=2] position = np.empty((iterations, initialPosition.size))
        np.ndarray[double, ndim=2] velocity = np.empty((iterations, initialVelocity.size))
        np.ndarray[double, ndim=1] currentAcceleration = np.empty(masses.size*2)
        np.ndarray[double, ndim=1] nextAcceleration = np.empty(masses.size*2)
        int i, j

    position[0] = initialPosition
    velocity[0] = initialVelocity
    
    for i in range(bodiesNumber):
        for d in range(2):
            currentAcceleration[i*2+d] = acceleration(i, masses, position[0])[d]
    
    for j in range(iterations - 1):
        position[j+1] = position[j] + velocity[j] * deltaTime + 0.5 * currentAcceleration * deltaTime * deltaTime
        
        for i in range(bodiesNumber):
            for d in range(2):
                nextAcceleration[i*2 + d] = acceleration(i, masses, position[j+1])[d]
            
        velocity[j+1] = velocity[j] + 0.5 * (currentAcceleration + nextAcceleration) * deltaTime
        currentAcceleration = nextAcceleration

    return position, velocity, times