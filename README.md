## LEARNING SUBJECT TO AVERAGE INDIVIDUAL FAIRNESS (AIF)
See our paper: https://arxiv.org/abs/1905.10607

## CITATION
The algorithm is implemented by modifying the expgrad reduction algorithm
(Agarwal et al. 2018, written in classred.py file) which is publicly available at: 
https://github.com/microsoft/fairlearn.
We therefore thank Microsoft Corporation for making their codes accessible.
We also thank the UCI Machine Learning Repository for the Communities and Crime 
data set available at: http://archive.ics.uci.edu/ml/datasets/communities+and+crime


## REQUIRED LIBRARIES
Our implementation requires the following libraries be already installed on python:
__future__, argparse, functools, numpy, pandas, 
clean_data, pickle, sklearn, scipy, collections


# ALGORITHM'S DESCRIPTION
Our algorithm AIF-Learn is implemented in 'aif_learn.py'. Here's a short description:

'aif_learn.py' takes as input:

	-d: data set (must be stored under ~/dataset, 
			currently valid inputs are: communities and synthetic)
	-alpha: allowed fairness violation (w.r.t. the center gamma)
	-max_iter: maximum number of iterations of the algorithm (default: 1500)
	-nu: desired duality gap of the output model (default: 0.001)

and outputs a dictionary stored as a pickle file under ~/pickles containing:

	-'err': the overall error of the learned fair model.
	-'gammahat': gammahat around which individual error rates concentrate.
	-'unfairness': unfairness of the learned model measured w.r.t. gammahat.
	-'unfairness_prime': maximum disparity between any two individual error.
	-'err_problem': error rate of each problem, averaged over individuals.
	-'err_individual': error rate of each individual, averaged over problems.
	-'error_t': the trajectory of overall error over time t.
	-'gamma_t': the trajectory of unfairness (w.r.t. gammahat) over time t.
	-'alpha': the alpha input to the algorithm.
	-'weight_set': the set of learned weights used to define the learned mapping.
	-'individuals': the set of training individuals used to define the mapping.


# AN EXAMPLE
Here is a fast toy example:

	python3 aif_learn.py -d communities -alpha 0.1 -max_iter 5


# VISUALIZATION
'plot_convergence.py' and 'plot_errorspread.py' visualize the results as in our paper,
and save the figures under ~/figures.


# SYNTHETIC DATA GENERATION
'synthetic_data_generator.py' generates synthetic data sets as described in the
supplement of our paper, and stores them under ~/dataset.


# SAVED FILES
We have our experimental results on the Communities and Crime as well as the
synthetic data set saved under ~/pickles. These are the outputs of our algorithm on 
the specified range of alpha after 1000 iterations. Corresponding figures appear
under ~/figures.
