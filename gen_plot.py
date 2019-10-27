# Plot generalization results
import pickle
import matplotlib.pyplot as plt

dataset = "communities"
n_train = 1595
m_train = 50
n_test = 399
m_test = 25

results = pickle.load( open("generalization_pickles/generalization_%(dataset)s_%(n_train)i_%(m_train)i_%(n_test)i_%(m_test)i.p" %locals(), "rb") )

training_err = results['training_err']
training_unf = results['training_unf']
test_err_individuals = results['test_err_individuals']
test_unf_individuals = results['test_unf_individuals']
test_err_problems = results['test_err_problems']
test_unf_problems = results['test_unf_problems']
test_err_both = results['test_err_both']
test_unf_both = results['test_unf_both']

plt.plot(training_err, training_unf, '.', color='red', label='training ($n_{train} = %(n_train)i, m_{train} = (m_train)i$)' %locals())
plt.plot(test_err_individuals, test_unf_individuals, '.', color='blue', label = r'gen. over individuals ($n_{test} = %(n_test)i$)' %locals())
plt.plot(test_err_problems, test_unf_problems, '.', color='green', label = r'gen. over problems ($m_{test} = %(m_test)i$)' %locals())
plt.plot(test_err_both, test_unf_both, '.', color='black', label = r'gen. over both ($n_{test} = %(n_test)i, m_{test} = %(m_test)i$)' %locals())

plt.plot(training_err, training_unf, color='red', linewidth=0.5)
plt.plot(test_err_individuals, test_unf_individuals, color='blue', linewidth=0.5)
plt.plot(test_err_problems, test_unf_problems, color='green', linewidth=0.5)
plt.plot(test_err_both, test_unf_both, color='black', linewidth=0.5)

plt.xlabel('error rate')
plt.ylabel(r'fairness violation (target for training: $2\alpha$)')
plt.legend(loc = 'best', prop={'size': 8})
plt.title(r'communities dataset (varying $\alpha=0.1,0.11, \ldots, 0.22$)')
#plt.savefig('figures/%(dataset)s_generalization.png' %locals(), dpi = 500)
plt.show()
