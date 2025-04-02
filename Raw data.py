import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

plt.show()