#Libraries
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import cyt
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import time
import random as random
import SecondTask as st
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue
from scipy.integrate import odeint


# a_i = d(v_i)/dt, where body has m_i, r_i, v_i
def acceleration(body, masses, positions):
    gravitationConstant = 6.67408e-11
    finalAcceleration = 0
    sunPosition = np.zeros((positions.shape[0], 2))
    
    for j in range(masses.size):
        if body != j:
            distance = positions[:, 2*j:2*j+2] - positions[:, 2*body:2*body+2] # r_j - r_i
            finalAcceleration += gravitationConstant * masses[j] * distance / np.linalg.norm(distance, 2)**3
    distance = sunPosition - positions[:, 2*body:2*body+2]
    finalAcceleration += gravitationConstant * 1.9 * pow(10, 30) * distance / np.linalg.norm(distance, 2) ** 3
    
    return finalAcceleration


def accelerationWithoutTime(body, masses, positions):
    gravitationConstant = 6.67408e-11
    finalAcceleration = 0    
    sunPosition = np.array([0.0, 0.0])
    
    for j in range(masses.size):
        if body != j:
            distance = positions[2*j:2*j+2] - positions[2*body:2*body+2] # r_j - r_i
            finalAcceleration += gravitationConstant * masses[j] * distance / np.linalg.norm(distance, 2)**3
    distance = sunPosition - positions[2*body:2*body+2]
    finalAcceleration += gravitationConstant * 1.9 * pow(10, 30) * distance / np.linalg.norm(distance, 2) ** 3
    
    return finalAcceleration


# # The same function as above, but without numpy operations with arrays
# def accelerationWithoutNumpy(body, masses, positions):
#     gravitationConstant = 6.67408e-11
#     finalAcceleration = 0
    
#     distance = np.zeros(2)
    
#     for j in range(masses.size):
#         if body != j:
#             for d in range(2):
#                 distance[d] = positions[2*j+d] - positions[2*body+d] # r_j - r_i
#                 finalAcceleration += gravitationConstant * masses[j] * distance / np.linalg.norm(distance, 2)**3
    
#     return finalAcceleration


# In[2]:


# Just Pyhton usage
def verletAlgorythm(masses, initialPosition, initialVelocity, deltaTime, iterations):
    times = np.arange(iterations) * deltaTime
    
    positions = np.zeros((times.size, 2*masses.size))
    positions[0] = initialPosition
    
    velocities = np.zeros((times.size, 2*masses.size))
    velocities[0] = initialVelocity
    
    currentAcceleration = np.zeros(2*masses.size)
    for i in range(masses.size):
        currentAcceleration[2*i:2*i+2] = acceleration(i, masses, positions)[0]  
        
    nextAcceleration = np.zeros(2*masses.size)

    for j in range(iterations-1):
        positions[j+1,:] = positions[j,:] + velocities[j,:] * deltaTime + 0.5 * currentAcceleration * deltaTime**2
        for i in range(masses.size):
            nextAcceleration[2*i:2*i+2] = acceleration(i, masses, positions)[j]
        velocities[j+1,:] = velocities[j,:] + 0.5 * (currentAcceleration + nextAcceleration) * deltaTime
        currentAcceleration = nextAcceleration

    return positions, velocities, times


# In[3]:



# odeint usage
def odeintVerletAlgorythm(masses, initialPosition, initialVelocity, deltaTime, iterations):
    times = np.arange(iterations) * deltaTime
    
    def system(posvel, t):
        res = np.zeros(posvel.shape)
        mid_idx = posvel.size // 2
        pos = posvel[:mid_idx]
        vel = posvel[mid_idx:]
        for i in range(masses.size):
            res[mid_idx+2*i:mid_idx+2*i+2] = accelerationWithoutTime(i, masses, pos)
        res[:mid_idx] = vel
        return res
    
    resultposvel = odeint(system, np.ndarray.flatten(np.concatenate((initialPosition, initialVelocity))), times)
    positions = resultposvel[:, :masses.size*2]
    velocities = resultposvel[:, masses.size*2:]
    
    return positions, velocities, times
        
        


# OpenCL usage
def openCLVerletAlgorithm(masses, initialPosition, initialVelocity, deltaTime, iterations):
    os.environ['PYOPENCL_CTX'] = '0'
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    
    times = np.arange(iterations) * deltaTime
    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    openCLMasses = np.array(masses, dtype=np.float64)
    massesBuffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=openCLMasses)
    
    openCLInitialPosition = np.array(initialPosition, dtype=np.float64)
    initialPositionBuffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=openCLInitialPosition)
    
    openCLInitialVelocity = np.array(initialVelocity, dtype=np.float64)
    initialVelocityBuffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=openCLInitialVelocity)
    
    openCLDeltaTime = np.array(deltaTime, dtype=np.float64)
    deltaTimeBuffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=openCLDeltaTime)
        
    openCLIterations = np.array(iterations, dtype=np.int64)
    iterationsBuffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=openCLIterations)
     
    openCLPosition = np.zeros((times.size, initialPosition.size), dtype=np.float64)
    positionBuffer = cl.Buffer(ctx, mf.WRITE_ONLY, openCLPosition.nbytes)
    
    openCLVelocity = np.zeros((times.size, initialVelocity.size), dtype=np.float64)
    velocityBuffer = cl.Buffer(ctx, mf.WRITE_ONLY, openCLVelocity.nbytes)
    
    openCLBodiesNumber = np.array(masses.size, dtype=np.int64)
    bodiesNumberBuffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=openCLBodiesNumber)
    
    openCLAccelerations = np.zeros((times.size, initialPosition.size), dtype=np.float64)
    accelerationsBuffer = cl.Buffer(ctx, mf.WRITE_ONLY, openCLAccelerations.nbytes)
        
    openCLTemporaryArray = np.zeros(2, dtype=np.float64)
    temporaryArrayBuffer = cl.Buffer(ctx, mf.WRITE_ONLY, openCLTemporaryArray.nbytes)
    
    prg = cl.Program(ctx,
                     """
                     void acceleration(__global float *positions, __global float *openCLMasses, __global float *openCLAccelerations, __global float *openCLTemporaryArray, int bodiesNumber, int body, int time)
                     {
                         float gravitationConstant = 6.67408;
                         float norm = 0.0;
                         float distance = 0.0;
                         int shift = time * 2;
                         
                         for (int d = 0; d < 2 * bodiesNumber; d++)
                         {
                             openCLAccelerations[shift + d] = 0.0;
                         }
                               
                         for (int j = 0; j < bodiesNumber; j++)
                         {
                                 norm = 0.0; 
                                 
                                 if (body != j)
                                 {
                                   
                                    
                                    for (int d = 0; d < 2; d++)
                                    {
                                        distance = positions[shift + j*2 + d] - positions[shift + body*2 + d];
                                        norm += pown(distance, 2);
                                        openCLTemporaryArray[d] = gravitationConstant * openCLMasses[j] * distance;
                                        
                                    }
                                    
                                    norm = pow(norm, 1.5f);
                                    
                                    for (int d = 0; d < 2; d++)
                                    {
                                        if (norm != 0)
                                        {
                                            openCLAccelerations[shift + body*2 + d] += openCLTemporaryArray[d] / norm;
                                        }
                                    }   
                                 }
                                 
                            }
                                                
                     }
                     
                     __kernel void verletAlgorithm(__global float *openCLMasses, __global float *openCLInitialPosition, __global float *openCLInitialVelocity,  __global float *openCLDeltaTime, __global int *openCLIterations, __global float *openCLPosition, __global float *openCLVelocity, __global int *openCLBodiesNumber, __global float *openCLAccelerations, __global float *openCLTemporaryArray)
                     {
                        int bodiesNumber = *openCLBodiesNumber, iterations = *openCLIterations;
                        float deltaTime = *openCLDeltaTime;
                        int shift = bodiesNumber * 2;
                        
                        for (int coordinate = 0; coordinate < bodiesNumber * 2; coordinate++)
                        {
                            openCLPosition[coordinate] = openCLInitialPosition[coordinate];
                            openCLVelocity[coordinate] = openCLInitialVelocity[coordinate];
                        }
                        
                        for (int j = 0; j < bodiesNumber; j++)
                        {
                            acceleration(openCLPosition, openCLMasses, openCLAccelerations, openCLTemporaryArray, bodiesNumber, j, 0);
                        }
                        
                        for (int t = 0; t < iterations; t++)
                        {
                        
                            for (int n = 0; n < bodiesNumber; n++)
                            {
                                for (int d = 0; d < 2; d++)
                                {
                                    openCLPosition[(t+1)*shift + n*2 + d] = openCLPosition[t*shift + n*2 + d] + openCLVelocity[t*shift + n*2 + d] * deltaTime + 0.5 * openCLAccelerations[t*shift + n*2 + d] * pown(deltaTime,2);
                                }
                            }
                            
                            for (int j = 0; j < bodiesNumber; j++)
                            {
                                acceleration(openCLPosition, openCLMasses, openCLAccelerations, openCLTemporaryArray, bodiesNumber, j, t+1);
                            }
                            
                            for (int n = 0; n < bodiesNumber; n++)
                            {
                                for (int d = 0; d < 2; d++)
                                {
                                    openCLVelocity[(t+1)*shift + n*2 + d] = openCLVelocity[t*shift + n*2 + d] + 0.5 * (openCLAccelerations[n*2 + d] + openCLAccelerations[shift + n*2 + d]) * deltaTime; 
                                }
                            }
 
                        }
                     }
                     """)
    try:
        prg.build()
    except:
        print("Error:")
        print(prg.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
        raise

    prg.verletAlgorithm(queue, (1,), None, massesBuffer, initialPositionBuffer, initialVelocityBuffer, deltaTimeBuffer, iterationsBuffer, positionBuffer, velocityBuffer, bodiesNumberBuffer, accelerationsBuffer, temporaryArrayBuffer)
    
    cl.enqueue_copy(queue, openCLPosition, positionBuffer).wait()
    cl.enqueue_copy(queue, openCLVelocity, velocityBuffer).wait()
    
    return openCLPosition, openCLVelocity, times


def accelerationForBody(body, masses, position):
    gravitationConstant = 6.67408e-11
    finalAcceleration = 0
    
    for j in range(masses.size):
        if j != body:
            distance = position[2*j:2*j+2]-position[2*body:2*body+2]
            finalAcceleration += gravitationConstant * masses[j] * distance /  np.linalg.norm(distance, 2)**3

    return finalAcceleration, body


def nextPosition(previousPosition, previousVelocity, previousAcceleration, masses, deltaTime, body):
    resultPosition = previousPosition[2*body:2*body+2] + previousVelocity[2*body:2*body+2] * deltaTime + 0.5 * previousAcceleration[2*body:2*body+2] * deltaTime**2
    return resultPosition, body


def nextVelocity(previousPosition, previousVelocity, previousAcceleration, masses, deltaTime, body):
    nextAcceleration = accelerationForBody(body, masses, previousPosition)
    resultVelocity = previousVelocity[2*body:2*body+2] + 0.5 * (nextAcceleration[0] + previousAcceleration[0]) * deltaTime    
    return resultVelocity, nextAcceleration, body


def MultiprocessingVerletAlgorythm(masses, initialPosition, initialVelocity, deltaTime, iterations):
    times = np.arange(iterations) * deltaTime
    
    startTime = time.time()

    initialAcceleration = np.zeros(2*masses.size)

    with ProcessPoolExecutor(max_workers = mp.cpu_count()) as executor:
        
        jobs = []
        for i in range(masses.size):
            jobs.append(executor.submit(accelerationForBody, body = i, masses = masses, position = initialPosition))       
        for job in as_completed(jobs):
            result_done = job.result()
            i = result_done[1]
            initialAcceleration[2*i:2*i+2] = result_done[0]
    
        for t in range(iterations-1):
            
            jobs = []
            for i in range(masses.size):
                jobs.append(executor.submit(nextPosition, previousPosition = initialPosition, previousVelocity = initialVelocity, previousAcceleration = initialAcceleration, masses = masses, deltaTime = deltaTime, body = i))            
            for job in as_completed(jobs):
                result_done = job.result()
                i = result_done[1]
                initialPosition[2*i:2*i+2] = result_done[0]
            
            jobs = []
            for i in range(masses.size):
                jobs.append(executor.submit(nextVelocity, previousPosition = initialPosition, previousVelocity = initialVelocity, previousAcceleration = initialAcceleration, masses = masses, deltaTime = deltaTime, body = i ))            
            for job in as_completed(jobs):
                result_done = job.result()
                i = result_done[2]
                initialVelocity[2*i:2*i+2] = result_done[0]
                initialAcceleration[2*i:2*i+2] = result_done[1][0]


    endTime = time.time()
    return initialPosition, initialVelocity, times, endTime - startTime


def VerletAlgorithmMethods(method, masses, initialPosition, initialVelocity, deltaTime, iterations):
    
    methods = {0: odeintVerletAlgorythm,
               1: verletAlgorythm,
               2: MultiprocessingVerletAlgorythm,
               3: cyt.cythonVerletAlgorithm,
               4: openCLVerletAlgorithm}
    
    return methods[method](masses, initialPosition, initialVelocity, deltaTime, iterations)





