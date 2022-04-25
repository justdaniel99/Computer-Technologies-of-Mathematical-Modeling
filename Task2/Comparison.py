#!/usr/bin/env python
# coding: utf-8

# In[43]:


import time
import numpy as np
import random as random
import SecondTask as st
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import cyt
from scipy.integrate import odeint


# In[44]:


def MultiprocessingVerletAlgorithm(masses, initialPosition, initialVelocity, deltaTime, iterations):
    if __name__ == '__main__':
        pool = mp.Pool(mp.cpu_count())
        before = time.time()
        result = pool.starmap(st.verletAlgorythm, [(masses, initialPosition, initialVelocity, deltaTime, iterations)])
        after = time.time()
        print (after-before)


def timeToSolve(method, launchesNumber, masses, initialPosition, initialVelocity, deltaTime, iterations):
    if (method != 2):
        t = 0
            
        for i in range(launchesNumber):
            startTime = time.time()
            st.VerletAlgorithmMethods(method, masses, initialPosition, initialVelocity, deltaTime, iterations)
            t += time.time() - startTime
        
        return t / launchesNumber
#     else:
#         for i in range(launchesNumber):
#             MultiprocessingVerletAlgorithm(masses, initialPosition, initialVelocity, deltaTime, iterations)


# In[45]:


def particlesGeneration(n, dx=1e12):
    
    def oneParticleGeneration(centre):
        m = random.uniform(1, 1e5) * 1e22
        r_x = random.uniform(-2*dx/3, 2*dx/3)
        r_y = random.uniform(-2*dx/3, 2*dx/3)
        v_x = random.uniform(-dx**0.3, dx**0.3)
        v_y = random.uniform(-dx**0.3, dx**0.3)
        return [m, centre[0] + r_x, centre[1] + r_y, v_x, v_y]

    random.seed()
    currentCentre = np.zeros(2)
    particles = [oneParticleGeneration(currentCentre)]
    k = 1
    for i in range(n // 2):
        currentCentre[0] += k * dx * (-1)**k
        particles.append(oneParticleGeneration(currentCentre))
        currentCentre[1] += k * dx * (-1)**k
        particles.append(oneParticleGeneration(currentCentre))
        k += 1

    # Format for computation
    mass = []
    position = []
    velocity = []
    for particle in particles:
        mass += particle[:1]
        position += particle[1:3]
        velocity += particle[3:]

    return np.array(mass), np.array(position), np.array(velocity), 10 * abs((max(position) - min(position)) / max(velocity))


# In[46]:


def main():
    m, p, v, t = particlesGeneration(50)
    #print(timeToSolve(1, 3, m, p, v, t / 50, 50))
    #print(timeToSolve(3, 3, m, p, v, t / 50, 50))
    #print(timeToSolve(4, 3, m, p, v, t / 50, 50))
    
    #MultiprocessingVerletAlgorithm(m, p, v, t / 50, 50)
    #MultiprocessingVerletAlgorithm(m, p, v, t / 50, 50)
    #MultiprocessingVerletAlgorithm(m, p, v, t / 50, 50)

    

main()




