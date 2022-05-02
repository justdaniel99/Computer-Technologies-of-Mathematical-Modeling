import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.animation as animation
import SecondTask as st


def accuracy(method, masses, initialPosition, initialVelocity, deltaTime, iterations):
    idealPosition, idealVelocity, time = st.VerletAlgorithmMethods(0, masses, initialPosition, initialVelocity, deltaTime, iterations)
    actualPosition, actualVelocity, time = st.VerletAlgorithmMethods(method, masses, initialPosition, initialVelocity, deltaTime, iterations)
    return np.amax(np.abs(idealPosition - actualPosition)), np.amax(np.abs(idealVelocity - actualVelocity))


deltaTime = pow(10, 6) * 1.5
scale = pow(10, 12)
iterations = pow(10,3)

width = 5 * scale
height = 5 * scale

sunPosition = np.array([0.0, 0.0])

masses = np.array([1.0243, 0.87, 5.68, 18.986, 0.006419, 0.057974, 0.04869, 0.003302], dtype = np.float64) * pow(10, 26)

planetsNumber = 8

initialPositions = np.array([4.5, 0.0, 2.8, 0.0, 1.43, 0.0, 0.8, 0.0, 0.228, 0.0, 0.15, 0.0, 0.108, 0.0, 0.058, 0.0], dtype = np.float64) * scale
initialVelocities = np.array([0.0, 5.4, 0.0, 6.8, 0.0, 9.69, 0.0, 13.07, 0.0, 24.13, 0.0, 29.76, 0.0, 35.02, 0.0, 47.87], dtype = np.float64) * pow(10, 3)

solvingResult = st.odeintVerletAlgorythm(masses, initialPositions, initialVelocities, deltaTime, iterations)

fig = plt.figure()
ax = plt.axes(xlim=(-width, width + scale), ylim=(-height, height))
ax.set_facecolor((0, 0, 0.1))
plt.xlabel("x, billion kilometer")
plt.ylabel("y, billion kilometer")
plt.title("Solar System")
plt.style.use("dark_background")

sun_color = "yellow"
sun = plt.Circle(sunPosition, 0.001 * scale, fc = sun_color)
ax.add_patch(sun)

neptune_color = "royalblue"
uranus_color = "skyblue"
saturn_color = "wheat"
jupiter_color = "peru"
mars_color = "coral"
earth_color = "darkcyan"
venus_color = "whitesmoke"
mercury_color = "silver"

legend_elements = [lines.Line2D([0], [0], color = neptune_color, lw = 4),
                   lines.Line2D([0], [0], color = uranus_color, lw = 4),
                   lines.Line2D([0], [0], color = saturn_color, lw = 4),
                   lines.Line2D([0], [0], color = jupiter_color, lw = 4),
                   lines.Line2D([0], [0], color = mars_color, lw = 4),
                   lines.Line2D([0], [0], color = earth_color, lw = 4),
                   lines.Line2D([0], [0], color = venus_color, lw = 4),
                   lines.Line2D([0], [0], color = mercury_color, lw = 4),
                   lines.Line2D([0], [0], color = sun_color, lw = 4)]

ax.legend(legend_elements, ["Neptune",
                            "Uranus",
                            "Saturn",
                            "Jupiter",
                            "Mars",
                            "Earth",
                            "Venus",
                            "Mercury",
                            "Sun"])

neptune = plt.Circle((4.5 * scale, 0.0), 0.13 * scale, fc = neptune_color)
uranus = plt.Circle((2.8 * scale, 0.0), 0.16 * scale, fc = uranus_color)
saturn = plt.Circle((1.43 * scale, 0.0), 0.19 * scale, fc = saturn_color)
jupiter = plt.Circle((0.8 * scale, 0.0), 0.22 * scale, fc = jupiter_color)
mars = plt.Circle((0.228 * scale, 0.0), 0.04 * scale, fc = mars_color) #second parameter to play
earth = plt.Circle((0.15 * scale, 0.0), 0.1 * scale, fc = earth_color) #second parameter to play
venus = plt.Circle((0.108 * scale, 0.0), 0.07 * scale, fc = venus_color) #second parameter to play
mercury = plt.Circle((0.058 * scale, 0.0), 0.01 * scale, fc = mercury_color) #second parameter to play

planets = []

def init():
    neptune.center = (4.5 * scale, 0.0)
    uranus.center = (2.8 * scale, 0.0)
    saturn.center = (1.43 * scale, 0.0)
    jupiter.center = (0.8 * scale, 0.0)
    mars.center = (0.228 * scale, 0.0)
    earth.center = (0.15 * scale, 0.0)
    venus.center = (0.108 * scale, 0.0)
    mercury.center = (0.058 * scale, 0.0)
    
    ax.add_patch(neptune)
    ax.add_patch(uranus)
    ax.add_patch(saturn)
    ax.add_patch(jupiter)
    ax.add_patch(mars)
    ax.add_patch(earth)
    ax.add_patch(venus)
    ax.add_patch(mercury)
    
    planets.append(neptune)
    planets.append(uranus)
    planets.append(saturn)
    planets.append(jupiter)
    planets.append(mars)
    planets.append(earth)
    planets.append(venus)
    planets.append(mercury)
    
    return planets

def animate(i):
    j = i % iterations 
    if j >= iterations:
        SystemExit()
    for k in range(planetsNumber):
        planets[k].center = solvingResult[0][j][2*k:2*k+2]
    return planets


planetsAnimation = animation.FuncAnimation(fig,
                           animate,
                           init_func=init,
                           frames=720,
                           interval=20,
                           blit=True)

#planetsAnimation.save("example.gif")

plt.show()