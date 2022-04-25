#Libraries
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import cyt
from scipy.integrate import odeint

# a_i = d(v_i)/dt, where body has m_i, r_i, v_i
def acceleration(body, masses, positions):
    gravitationConstant = 6.67408e-11
    finalAcceleration = 0
    
    for j in range(masses.size):
        if body != j:
            distance = positions[:, 2*j:2*j+2] - positions[:, 2*body:2*body+2] # r_j - r_i
            finalAcceleration += gravitationConstant * masses[j] * distance / np.linalg.norm(distance, 2)**3
    
    return finalAcceleration


# The same function as above, but without numpy operations with arrays
def accelerationWithoutNumpy(body, masses, positions):
    gravitationConstant = 6.67408e-11
    finalAcceleration = 0
    
    distance = np.zeros(2)
    
    for j in range(masses.size):
        if body != j:
            for d in range(2):
                distance[d] = positions[2*j+d] - positions[2*body+d] # r_j - r_i
                finalAcceleration += gravitationConstant * masses[j] * distance / np.linalg.norm(distance, 2)**3
    
    return finalAcceleration


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
        
    def problemPosition(v, t):
        res = np.zeros(v.size)
        for i in range(v.size):
            res[i] = v[i]
        return res
    
    def problemVelocity(x, t):
        res = np.zeros(x.size)
        
        positions = np.zeros((times.size, x.size))
        positions[0] = x
        
        for i in range(x.size//2):
            res[2*i:2*i+2] = acceleration(i, masses, positions)[i]
        return res

    positions = odeint(problemPosition, initialPosition, times)
    velocities = odeint(problemVelocity, initialVelocity, times)
    
    return positions, velocities, times


# In[4]:


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
    
    cl.enqueue_copy(queue, positionBuffer, openCLPosition).wait()
    cl.enqueue_copy(queue, velocityBuffer, openCLVelocity).wait()
    
    return openCLPosition, openCLVelocity, times


# Multiprocessing usage
def MultiprocessingVerletAlgorithm(masses, initialPosition, initialVelocity, deltaTime, iterations):
    if __name__ == '__main__':
        before = time.time()
        pool = multiprocessing.Pool(mp.cpu_count())
        result = pool.starmap(verletAlgorythm, [(masses, initialPosition, initialVelocity, deltaTime, iterations)])
        after = time.time()
        print (after-before)
        
#     if __name__ == '__main__':
#         with mp.Pool(5) as pool:
#             pool.starmap(verletAlgorythm, [(masses, initialPosition, initialVelocity, deltaTime, iterations)])
#     global verlet
    
#     def verlet(masses, initialPosition, initialVelocity, deltaTime, iterations):
#         positions, velocities, times = verletAlgorythm(fucks, initialPosition, initialVelocity, deltaTime, iterations)
#         return positions, velocities, times

 
    

# Multiprocessing usage
# def MultiprocessingVerletAlgorithm(masses, initialPosition, initialVelocity, deltaTime, iterations): 
#     global verlet
    
#     def verlet(queue, resultQueue, initialVelocity, sharedMemoryPositions, body, events1, events2):
        
#         currentPosition = np.array(sharedMemoryPositions[body*2:(body+1)*2])
#         currentAcceleration = accelerationWithoutNumpy(body, masses, sharedMemoryPositions)

#         resultPosition = np.empty((iterations, 2))
#         resultVelocity = np.empty((iterations, 2))
        
#         resultPosition[0, :] = currentPosition
#         resultVelocity[0, :] = initialVelocity[body*2:(body+1)*2]
    
#         for j in range(iterations - 1):

#             resultPosition[j+1, :] = currentPosition + resultVelocity[j, :] * deltaTime + 0.5 * currentAcceleration * deltaTime**2

#             queue.put([body, resultPosition[j+1, :]])
            
#             events1[body].set()

#             if body == 0:
#                 for i in range(masses.size):
#                     events1[i].wait()
#                     events1[i].clear()
                    
#                 for i in range(masses.size):
#                     gettingFromQueue = queue.get()
#                     sharedMemoryPositions[gettingFromQueue[0]*2:(gettingFromQueue[0]+1)*2] = gettingFromQueue[1]
                    
#                 for i in range(masses.size):
#                     events2[i].set()
#             else:
#                 events2[body].wait()
#                 events2[body].clear()

#             currentPosition = np.array(sharedMemoryPositions[body*2:(body + 1)*2])
            
#             nextAcceleration = accelerationWithoutNumpy(body, masses, sharedMemoryPositions)

#             resultVelocity[j+1, :] = resultVelocity[j, :] + 0.5 * (currentAcceleration + nextAcceleration) * deltaTime
            
#             currentAcceleration = nextAcceleration
            
#         resultQueue.put([body, resultPosition, resultVelocity])

    
#     if __name__ == '__main__':
#         times = np.arange(iterations) * deltaTime
        
#         events1 = []
#         events2 = []
#         processes = []
        
#         positions = np.zeros((times.size, 2*masses.size))
#         velocities = np.zeros((times.size, 2*masses.size))
        
#         sharedMemoryPositions = mp.Array('d', initialPosition)

#         for body in masses:
#             events1.append(mp.Event())
#             events2.append(mp.Event())
#             events1[-1].clear()
#             events2[-1].clear()

#         queue = mp.Queue()
#         resultQueue = mp.Queue()
        
#         for body in range(masses.size):
#             processes.append(mp.Process(target=verlet, args=(queue, resultQueue, initialVelocity, sharedMemoryPositions, body, events1, events2)))
#             processes[-1].start()

#         for i in range(masses.size):
#             gettingFromQueue = resultQueue.get()
#             positions[:, gettingFromQueue[0]*2:(gettingFromQueue[0]+1)*2] = gettingFromQueue[1]
#             velocities[:, gettingFromQueue[0]*2:(gettingFromQueue[0]+1)*2] = gettingFromQueue[2]

#         for process in processes:
#             process.join()

#         return positions, velocities, times


def VerletAlgorithmMethods(method, masses, initialPosition, initialVelocity, deltaTime, iterations):
    
    methods = {0: odeintVerletAlgorythm,
               1: verletAlgorythm,
               2: MultiprocessingVerletAlgorithm,
               3: cyt.cythonVerletAlgorithm,
               4: openCLVerletAlgorithm}
    
    return methods[method](masses, initialPosition, initialVelocity, deltaTime, iterations)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




