# documentation
# results visualization: error spread + comparison with the baseline
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
eta = 200
max_iter = 1000
nu = 0.001
alpha_list = [i/100 for i in range(2,23)] # the range of alpha values to plot
gap = 0.01 # the plot for the base line model is shifted this much upward to avoid occlusion
#######################################

alpha_max = max(alpha_list) # the alpha for the unconstrained learning problem
results_unconstrained = pickle.load( open("old_pickles/%(dataset)s_%(max_iter)i_%(alpha_max).2f_%(nu).3f_aif.p" %locals(), "rb") )
individual_err_vals_unconstrained = list(results_unconstrained['err_individual'].values())
err_val_unconstrained = results_unconstrained['err']

errval = []
unfval = []
errval_mix = []
unfval_mix = []

for alpha in alpha_list:
    results = pickle.load( open("old_pickles/%(dataset)s_%(max_iter)i_%(alpha).2f_%(nu).3f_aif.p" %locals(), "rb") )
    individual_err_vals = list(results['err_individual'].values())
    err_val = results['err']
    ############################
    errval.append(err_val)
    unfval.append(2*alpha)
    ############################
    alpha_mix = alpha/alpha_max
    individual_err_vals_mix = [alpha_mix * i + 0.5*(1-alpha_mix) for i in individual_err_vals_unconstrained]
    err_val_mix = alpha_mix * err_val_unconstrained + 0.5*(1-alpha_mix)
    ############################
    errval_mix.append(err_val_mix)
    unfval_mix.append(2*alpha+0.01)
    ############################
    if alpha != alpha_list[len(alpha_list)-1]:
        plt.plot(individual_err_vals, [2*alpha]*n, '.', color = 'royalblue')
        plt.plot(err_val, [2*alpha], '.', color = 'red')
        plt.plot(individual_err_vals_mix, [2*alpha+gap]*n, '.', color = 'grey')
        plt.plot(err_val_mix, [2*alpha+gap], '.', color = 'black')
    if alpha == alpha_list[len(alpha_list)-1]:
        plt.plot(individual_err_vals_mix, [2*alpha]*n, '.', color = 'royalblue', label='individual error rate')
        plt.plot(err_val, [2*alpha], '.', color = 'red', label = 'overall error rate')
        plt.plot(individual_err_vals, [2*alpha+gap]*n, '.', color = 'grey', label='individual error rate \n of the baseline model')
        plt.plot(err_val_mix, [2*alpha+gap], '.', color = 'black', label='overall error rate \n of the baseline model')


plt.plot(errval, unfval, color='red', linewidth=0.5)
plt.plot(errval_mix, unfval_mix, color='black', linewidth=0.5)
plt.xlabel('error rate', fontsize = 12)
plt.ylabel(r'$2\alpha$: allowed fairness violation')
plt.legend(loc = 'best', prop={'size': 11})
plt.yticks(np.arange(2*min(alpha_list), 2*max(alpha_list)+0.02, 0.04))
plt.title(r'error spread: %(dataset)s ($n=%(n)i, \, m=%(m)i, \, d=%(d)i$)' %locals())
#plt.savefig('figures/%(dataset)s_errspread.png' %locals(), dpi = 500)
plt.show()
