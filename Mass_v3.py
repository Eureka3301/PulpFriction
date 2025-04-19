# Differential equation of motion for a given displacement $u$:
# 
# $m*\ddot{x} = k*(u-x) - kh*x - 2 \gamma*\dot{x} \mp f$,
# where sign is determined by motion direction and
# $f$ is static or sliding friction (with coefficient $\mu$).


# importing nessecary libraries

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

# my lovely printing function

import os
def prnt(s=''):
    if s:
        print(s)
    print('='*os.get_terminal_size()[0])

# creating class of single mass on spring with given parameters

g = 9.81 # m/s2

class model():
    def __init__(self,
                 m, k, kh, mu, gamma,
                 finess = 100,
                 **kwargs):
        
        # creating timeline
        self.t = 0
        # setting initial conditions
        self.x = 0
        self.v = 0
        # setting boundary condition
        self.u = 0

        # setting the oscillator parameters
        self.m = m # kg
        self.k = k # N/m
        self.kh = kh # N/m
        self.mu = mu
        self.gamma = gamma # N / m/s
        
        # sliding friction threshold
        self.fric = self.mu*self.m*g

        # computing timestep for integration
        self.omega = np.sqrt(self.k/self.m)
        self.dt = 2*np.pi/self.omega / finess

        prnt()
        print('Mass has been created:')
        print(f'm = {self.m} kg, k = {self.k} N/m, kh = {self.kh} N/m', end=', ')
        print(f'mu = {self.mu}, gamma = {self.gamma} N/m')
        prnt()
        print(f't = {self.t}, x(t) = {self.x}, v(t) = {self.v}, u(t) = {self.v}')
        print(f'dt = {self.dt:.1e} 1/s')

    # calculating the friction
    def friction(self):
        # guess the moving direction
        drct = np.sign(self.v)

        # define current friction
        if drct:
            f = drct*self.fric
        else:
            # estimate external force
            resForce = self.k*(self.u-self.x) - self.kh*self.x - self.gamma*self.v
            f = resForce
        return f

    # calculating new state after given increment du over time step dt
    def step_integrate(self, du):
        # define current derivative of displacement
        dotu = du/self.dt

        # guess the moving direction
        drct = np.sign(self.v)

        # define current friction and its derivative
        if drct:
            f = drct*self.fric
            dotf = 0
        else:
            # estimate external force
            resForce = self.k*(self.u-self.x) - self.kh*self.x - self.gamma*self.v
            f = resForce
            dotf = self.k*(self.u)
        
        # define current acceleration and its derivative
        a = self.k*(self.u-self.x)-self.kh*self.x-2*self.gamma*self.v-f
        dota = self.k*(dotu-self.v)-self.kh*self.v-2*self.gamma*a-dotf

        # assemble new state of mass

        self.x += self.v*self.dt + a * self.dt*self.dt/2
        self.v += a * self.dt + dota * self.dt*self.dt/2
        self.u += du
        self.t += self.dt

        prnt(f't={self.t}, x={self.x}, v={self.v}, u={self.u}, f={f}')

        return (self.t, self.x, self.v, self.u, f)

    def visualise(self, tt, xx, vv, uu, ff):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            num='Visualisation',
            figsize = (13,7),
            nrows = 2, ncols = 2,
            constrained_layout = True,
            )

        ax1.set_xlabel('time, s')
        ax1.set_ylabel('coordinate, m')
        ax1.plot(tt, uu, label='pull')
        ax1.plot(tt, xx, label='mass')
        ax1.legend()

        ax2.set_xlabel('time, s')
        ax2.set_ylabel('velocity, m/s')
        ax2.plot(tt, vv)
        ax2.hlines(0, tt[0], tt[-1], color='r')

        ax3.set_xlabel('pull displacement, m')
        ax3.set_ylabel('spring stress, m')
        ax3.plot(uu, self.k*(uu-xx))

        ax4.set_xlabel('time, s')
        ax4.set_ylabel('friction, N')
        ax4.plot(tt, ff)
        ax4.hlines(self.fric, tt[0], tt[-1], color='r')

        ax2.sharex(ax4)

        plt.show()

    # calculating new state after all increment steps=([dt1, dt2,...], [du1, du2,...])
    def integrate(self, steps):
        
        # preparing history arrays
        tt = np.array([self.t])
        xx = np.array([self.x])
        vv = np.array([self.v])
        uu = np.array([self.u])
        ff = np.array([self.friction()])
        
        # unpacking steps
        dt = steps[0]
        du = steps[1]

        # view number of increment steps
        stepNum = len(dt)
        for n in range(stepNum):
            if dt[n] < self.dt:
                print(f'too small timestep for displacement change',end=', ')
                print(f'{dt[n]} < {self.dt}')
                prnt()
                return (tt, xx, vv, uu, ff)
            
            # define displacement derivative at current step
            dotu = du[n]/dt[n]
            
            # step by step, bit by bit
            t_start = self.t
            while self.t - t_start < dt[n]:
                (t, x, v, u, f) = self.step_integrate(dotu*self.dt)

                # remember the result
                tt=np.append(tt, t)
                xx=np.append(xx, x)
                vv=np.append(vv, v)
                uu=np.append(uu, u)
                ff=np.append(ff, f)
                
            prnt(f'dt_{n} = {dt[n]}, du_{n} = {du[n]}')

        # visualise
        self.visualise(tt, xx, vv, uu, ff)

        return (tt, xx, vv, uu, ff)
