import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

g = 9.81 # m/s2

class setup():
### initialising independent and dependent parameters of the system
    def __init__(self, **kwargs):
        self.m = kwargs['m'] # kg
        self.k = kwargs['k'] # N/m
        self.r = kwargs['r'] # m/s
        self.mu = kwargs['mu']
        self.nu = kwargs['nu']
        self.gamma = kwargs['gamma'] # N / m/s
        # just esteemating the natural frequency of the spring oscillator
        self.omega0 = np.sqrt(self.k/self.m) # 1/s
        # drag starts when spring is stronger than static friction
        self.t0 = self.nu*self.m*g/self.k/self.r # s

        print('------------setup calculated parameters------------')
        print(f'drag start = {self.t0} s, natural frequency = {self.omega0} 1/s')

### integrating the differencial equatian of motion with Taylor series expansion up to second order derivatives
    def __integrate(self, t_stop, timeFinement = 100):
        dt = 1/self.omega0/timeFinement
        
        print('------------integrating parameters------------')
        print(f't_stop = {t_stop} s, timeFinement = {timeFinement}, dt = {dt} s')

        if t_stop - self.t0 < dt:
            # just giving zeros, the static friction holds the mass
            tt = np.arange(0, t_stop, dt)
            xx = np.zeros_like(tt)
            vv = np.zeros_like(tt)
            return (tt, xx, vv, self.k*self.r*tt)
        
        else:
            # prepare the grid, velocity grid is needed because the mothion eq is second order ODE
            tt = np.arange(0, t_stop, dt)
            xx = np.zeros_like(tt)
            vv = np.zeros_like(tt)
            # total amount of steps
            stepNum = len(xx)
            # evaluating the step, when the drag starts (tt[num0-1] < t0 and tt[num0] >= t0)
            num0 = int(np.ceil(self.t0/dt))
            # friction force for interest, till t0 compensate the spring
            ff = np.zeros_like(tt)
            for step in range(num0):
                ff[step] = self.k*self.r*tt[step]
            # starting manual integration
            for step in range(num0, stepNum):
                dotv = (self.k*(self.r*tt[step-1]-xx[step-1])-ff[step-1])/self.m - 2*self.gamma/self.m*vv[step-1]
                ddotv = self.k*(self.r-vv[step-1])/self.m - 2*self.gamma/self.m*dotv
                vv[step] = vv[step-1] + dotv*dt + ddotv * dt*dt/2

                xx[step] = xx[step-1] + vv[step-1]*dt + dotv * dt*dt/2

                ff[step] = np.sign(vv[step]) * self.mu*self.m*g

            return (tt, xx, vv, ff)

    ### What all mechanics dream about
    def plot_analogy(self, t_stop, noSave=False, **kwargs):
        tt, xx, vv, ff = self.__integrate(t_stop, **kwargs)

        plt.ylabel('Spring stress, N')
        plt.xlabel('Spring movement, m')
        
        paramtext = f'm={self.m}\nk={self.k}\nr={self.r}\n$\\nu$={self.nu}\n$\mu$={self.mu}'
        plt.annotate(paramtext, xy=(1.015, 0.5), xycoords='axes fraction',
                     bbox=dict(facecolor='none', edgecolor='green', boxstyle='round'))

        plt.plot(self.r*tt, self.k*(self.r*tt-xx))

        if noSave == False:
            plt.savefig('output/analogy.jpg')



df_d = pd.read_csv('raw data\digitizer\Dynamic.csv',
                    names=['eps, %', 'stress, kgF/m2'],
                    delimiter=';', decimal=',')

df_s = pd.read_csv('raw data\digitizer\Static.csv',
                    names=['eps, %', 'stress, kgF/m2'],
                    delimiter=';', decimal=',')

g = 9.81 # m/s2
l0 = 5e-2 # m
d0 = 1e-2 # m
S0 = 3.14*d0*d0/4 # m2

df_s['dl, mm'] = df_s['eps, %']*10 * l0
df_s['F, N'] = df_s['stress, kgF/m2']*g * S0

df_d['dl, mm'] = df_d['eps, %']*10 * l0
df_d['F, N'] = df_d['stress, kgF/m2']*g * S0

n = 4* 200

for r in np.linspace(1e-3, 40e-2, 2):
    param = {
    'm' : 1e-2, # kg
    'k' : 2e-1, # N/m
    'r' : r, # m/s
    'mu' : 0.3,
    'nu' : 0.3,
    'gamma' : 1e-2, # N / m/s
    }

    s = setup(**param)
    s.plot_analogy(4/s.r, noSave=True)

sns.lineplot(df_s,
             x='dl, mm',
             y='F, N',
             label='dynamic',
)

sns.lineplot(df_d,
             x='dl, mm',
             y='F, N',
             label='static',
)

plt.savefig('output/match.jpg')