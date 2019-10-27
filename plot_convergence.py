# documentation
# results visualization: convergence and trajectories for error and unfairness
# figures are saved under ~/figures.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

############### Inputs ################
dataset = "communities"
n = 200 #sample size
m = 50 #problem size
d = 20 #dimenstion
max_iter = 1000
nu = 0.001
alpha_list = [0.05, 0.1, 0.15, 0.2] # the range of alpha values to plot
alpha_max = 0.22 # the alpha for the unconstrained learning problem
#######################################

results = pickle.load( open("pickles/%(dataset)s_%(max_iter)i_%(alpha_max).2f_%(nu).3f_aif.p" %locals(), "rb") )
unf_max = results['unfairness_prime']

for alpha in alpha_list:
    results = pickle.load( open("pickles/%(dataset)s_%(max_iter)i_%(alpha).2f_%(nu).3f_aif.p" %locals(), "rb") )
    error_t = results['error_t']
    gamma_t = [2*i for i in results['gamma_t']]
    gamma_t[0] = unf_max
    plt.scatter(error_t, gamma_t, s =20, c = range(max_iter))
    unfairness = 2*alpha
    plt.plot(error_t, gamma_t, label = r'$2\alpha =$ %.2f' %unfairness)
    plt.axhline(y=unfairness, linestyle='dashed', color = 'black')
    plt.annotate(r'$t=0$', (error_t[0], gamma_t[0]), verticalalignment='bottom',horizontalalignment='right')
    plt.annotate(r'$t=%(max_iter)i$' %locals(), (error_t[max_iter - 1], gamma_t[max_iter-1]), 
            verticalalignment='top',horizontalalignment='right')

plt.xlabel(r'$(error)_t$')
plt.ylabel(r'$(unfairness)_t$')
plt.xlim(0.1,0.5)
plt.legend(loc = 'best')
plt.title(r'convergence: %(dataset)s ($n=%(n)i, \, m=%(m)i, \, d=%(d)i$)' %locals())
plt.savefig('figures/%(dataset)s_convergence.png' %locals(), dpi = 500)
plt.show()
