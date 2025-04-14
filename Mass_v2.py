
# A mass $m$ on a spring with stiffness $k$ on a rougn surface (sliding friction $\mu$) with viscous dissipation $\gamma$.  


# Differential equation of motion for a given displacement $u$:
# 
# $\ddot{x} = \dfrac{k(u-x) - 2 \gamma \dot{x} \mp f}{m}$, where sign is determined by motion direction and $f$ is sliding friction $\mu m g$.


# ### Importing nessesary modules.


from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np


# ### Creating class of a single mass.


g = 9.81 # m/s2

class mass():
    def __init__(self, finess = 100, **kwargs):
        # setting initial conditions
        self.t = 0
        self.x = 0
        self.v = 0
        self.u = 0



        # setting the mass parameters
        self.m = kwargs['m'] # kg
        self.k = kwargs['k'] # N/m
        self.mu = kwargs['mu']
        self.gamma = kwargs['gamma'] # N / m/s
        
        # computing integrating interval
        self.omega = np.sqrt(self.k/self.m)
        self.dt = 2*np.pi/self.omega / finess

        print('Mass has been created:')
        print('===================================== parameters ======================================')
        print(f'mass = {self.m} kg, stiffnes = {self.k} N/m, mu = {self.mu}, gamma = {self.gamma} N/m, dt = {self.dt:.1e} 1/s')
        print('================================= initial conditions ==================================')
        print(f't = {self.t}, x(t) = {self.x}, v(t) = {self.v}')

    # moving check due to numerical error near v=0
    def is_moving(self):
        return np.sign(self.v)

    # calculating new state after given increment du over time step dt
    def step_integrate(self, du):
        # define current derivative of displacement
        dotu = du/self.dt
        self.t += self.dt

        # define current friction and its derivative
        if self.is_moving() == 0:
            f = self.is_moving()*np.min([np.abs(self.k*(self.u-self.x)), self.mu*self.m*g])
            dotf = self.k*dotu
        else:
            f = self.is_moving()*np.min([np.abs(self.k*(self.u-self.x)), self.mu*self.m*g])
            dotf = 0
        
        # define current acceleration and its derivative
        a = self.k*(self.u-self.x)-2*self.gamma*self.v-f
        dota = self.k*(dotu-self.v)-2*self.gamma*a-dotf

        # assemble new state of mass

        self.x += self.v*self.dt + a * self.dt*self.dt/2
        self.v += a * self.dt + dota * self.dt*self.dt/2
        self.u += du

        return f


    def integrate(self, t_steps, u_steps):
        
        tt = np.array([self.t])
        xx = np.array([self.x])
        vv = np.array([self.v])
        uu = np.array([self.u])
        ff = np.array([0])
        
        stepNum = len(t_steps)
        for step in range(1, stepNum):
            if t_steps[step] - t_steps[step-1] < self.dt:
                print(f'too small timestep for displacement change {t_steps[step] - t_steps[step-1]} < {self.dt}')
                return (tt, xx, vv, ff)
            dotu = (u_steps[step] - u_steps[step-1])/(t_steps[step] - t_steps[step-1])
            while self.t - tt[0] < t_steps[step]:
                f = self.step_integrate(dotu*self.dt)

                tt=np.append(tt, self.t)
                xx=np.append(xx, self.x)
                vv=np.append(vv, self.v)
                ff=np.append(ff, f)
                uu=np.append(uu, self.u)
        
            print(f't_step = {t_steps[step]}, u_step = {u_steps[step]}')

        return (tt, xx, vv, ff, uu)


params = {
    'm' : 50e-3, # kg
    'k' : 5, # N/m
    'mu' : 0.35,
    'gamma' : 0.5, # N/m
}

initial = {
    't0' : 0,
    'x0' : 0,
    'v0' : 0,
}

m = mass(**initial, **params)

r = 1e-1 # m/s

t_steps = np.linspace(0, 5, 2)
u_steps = r*t_steps


(tt1, xx1, vv1, ff1, uu1) = m.integrate(t_steps, u_steps)

r = -1e-1 # m/s

t_steps = np.linspace(0, 5, 2)
u_steps = r*t_steps

(tt2, xx2, vv2, ff2, uu2) = m.integrate(t_steps, u_steps)

tt = np.append(tt1, tt2)
xx = np.append(xx1, xx2)
vv = np.append(vv1, vv2)
ff = np.append(ff1, ff2)
uu = np.append(uu1, uu2)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize = (12,7), nrows = 2, ncols = 2)

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
ax3.plot(uu, m.k*(uu-xx))

ax4.set_xlabel('time, s')
ax4.set_ylabel('friction, N')
ax4.plot(tt, ff)
ax4.hlines(m.mu*m.m*g, tt[0], tt[-1], color='r')

ax2.sharex(ax4)

plt.show()


