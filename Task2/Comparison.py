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

def timeToSolve(method, launchesNumber, masses, initialPosition, initialVelocity, deltaTime, iterations):
     if (method != 2):
        t = 0
            
        for i in range(launchesNumber):
            startTime = time.time()
            st.VerletAlgorithmMethods(method, masses, initialPosition, initialVelocity, deltaTime, iterations)
            t += time.time() - startTime
        
        return t / launchesNumber


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


def accuracy(method, masses, initialPosition, initialVelocity, deltaTime, iterations):
    idealPosition, idealVelocity, time = st.VerletAlgorithmMethods(0, masses, initialPosition, initialVelocity, deltaTime, iterations)
    actualPosition, actualVelocity, time = st.VerletAlgorithmMethods(method, masses, initialPosition, initialVelocity, deltaTime, iterations)

    return np.amax(np.abs(idealPosition - actualPosition)), np.amax(np.abs(idealVelocity - actualVelocity))


# def main():
#     m, p, v, t = particlesGeneration(200)
#     print(timeToSolve(1, 3, m, p, v, t / 100, 100))
#     print(timeToSolve(3, 3, m, p, v, t / 100, 100))
#     print(timeToSolve(4, 3, m, p, v, t / 100, 100))


# main()

m, p, v, t = particlesGeneration(8)
print(type(m))
print(type(p))
print(type(v))
print(type(t))
print(accuracy(3, m, p, v, 10, 10))
# #print(st.odeintVerletAlgorythm(m, p, v, 5, 5))
  
    
# if __name__ == '__main__':
#     m, p, v, t = particlesGeneration(200)
#     print(st.MultiprocessingVerletAlgorythm(m, p, v, t / 100, 100)[3])
#     print(st.MultiprocessingVerletAlgorythm(m, p, v, t / 100, 100)[3])
#     print(st.MultiprocessingVerletAlgorythm(m, p, v, t / 100, 100)[3])

    






