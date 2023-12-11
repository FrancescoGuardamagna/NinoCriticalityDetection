import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

rc('axes', linewidth=4)
rc('font', weight='bold')
rcParams['xtick.major.width'] = 3
rcParams['xtick.major.size'] = 10
rcParams['ytick.major.width'] = 3
rcParams['ytick.major.size'] = 10
rcParams["font.weight"] = "bold"
rcParams["axes.labelweight"] = "bold"

class basic_bifurcation:

    def __init__(self,u,w,dt,noise_amplitude):

        self.u=u
        self.w=w
        self.dt=dt
        self.noise_amplitude=noise_amplitude

    def derivative(self,states):

        x=states[0]
        y=states[1]

        derivatives=[(self.u-x**2-y**2)*x+self.w*y,-self.w*x+(self.u-x**2-y**2)*y]

        return derivatives

    def integrate_euler_maruyama(self,steps,initial_conditions):

        solution=[]
        solution.append(initial_conditions)
        x0,y0=initial_conditions

        for i in range(steps):

            derivatives=self.derivative([x0,y0])
            x0=x0+derivatives[0]*self.dt+self.noise_amplitude*np.random.normal(loc=0,scale=1)*np.sqrt(self.dt)
            y0=y0+derivatives[1]*self.dt+self.noise_amplitude*np.random.normal(loc=0,scale=1)*np.sqrt(self.dt)
            solution.append([x0,y0])

        return np.array(solution)

    def plotting(self,initial_conditions,steps=10000,steps_skip=6000):

        solution=self.integrate_euler_maruyama(steps,initial_conditions)

        fig=plt.figure(figsize=(8,5))
        plt.plot(solution[steps_skip:,0],'-k',linewidth=2)
        plt.xlabel("steps",fontsize=18)
        plt.ylabel("x",fontsize=18)


    