##################################
#  Import required dependencies  #
##################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.stats import multivariate_normal
import pandas as pd
from MLE import MLEVAR, MLEVARsim, wprctile, process_VAR
import os
from tqdm import tqdm
import time
import sys
from numba import jit
from multiprocessing import Pool, current_process, Manager

# Set the options for printing numpy arrays neatly
np.set_printoptions(precision=3, legacy = '1.13')

###################################
#            LOAD DATA            #
###################################

T = pd.read_csv('data3py.csv')

date = T.date
popu = T.popu
propinc = T.propinc
corpprof = T.corpprof
pdivinc = T.pdivinc
consnond = T.consnond
consserv = T.consserv
pinond = T.pinond
piserv = T.piserv

start_date = date.iloc[0]
print("\nTime period:")
print("\tStart date: \t{}".format(start_date))

# This offset allows for a trunkated time period. Set to 0 to use all time
# periods
offset = 0

end_date = date.iloc[-offset - 1]
print("\tEnd date: \t{}".format(end_date))

# Filter out any observations which are outside the relevant time period
data_index = date.values <= end_date

CN      = consnond[data_index]                   # nondurables
CS      = consserv[data_index]                   # services
PCN     = pinond[data_index]/100                 # price index for nondurables
PCS     = piserv[data_index]/100                 # price index for services
NIPAE   = corpprof[data_index]                   # corporate before-tax profit with IVA and CCadj
NIPAPI  = propinc[data_index]                    # proprieters income
NIPAPDI = pdivinc[data_index]                    # personal dividend income
POP     = popu[data_index]                       # population series from LNU000000
PCE     = ((PCN*CN+PCS*CS)/(CN+CS))[data_index]  # weighted aggregate deflator
C       = (CN+CS)[data_index]                    # nominal consumption
c       = (C/PCE)[data_index]                    # real consumption
cpc     = (c/POP)[data_index].values             # real consumption per capita

e2       = ((NIPAE + NIPAPI)/PCE)[data_index]    # business income (proprietor's income plus corporate profits)[
e2pc     = (e2/POP)[data_index].values           # business income per capita

e3       = (NIPAPDI/PCE)[data_index]             # personal dividend income
e3pc     = (e3/POP)[data_index].values           # personal dividend income per capita


# We use the log values of the relevant variables
logcpc = np.log(cpc)
logepc = np.log(e2pc)
logdpc = np.log(e3pc)

# These are the actual variables we use in the VAR
gcpc = np.diff(logcpc)
logecpc = logepc - logcpc
logdcpc = logdpc - logcpc


####################################
#        SET UP TO RUN VAR         #
####################################

L = 5 # The number of lags used (consumption uses 1 less lag)
n = 3 # The dimension of the auto-regressive vector

# Load the vector of data
y = [] # The elements of y are of different lengths, so we leave it as a list
y.append(gcpc)
y.append(logecpc)
y.append(logdcpc)

# Code the initial observation X_0, used in Monte Carlo estimation
X0 = np.array([y[0][3], y[1][4], y[2][4],
     y[0][2], y[1][3], y[2][3],
     y[0][1], y[1][2], y[2][2],
     y[0][0], y[1][1], y[2][1],
     y[1][0], y[2][0]])

lags = [L-1, L, L] # Specify the number of lags used per variable
num_vars = np.sum(lags)
noncinds = np.arange(12).tolist() + np.arange(13,15).tolist()
# noncinds is used to drop the portions of the matrix A
# from the paper's Appendix B.1 that correspond to a fifth lag of
# consumption growth.

# Estimate the system as a VAR(5) model
B, S, y_lagged, X_lagged = MLEVAR(y, lags)

# Rearrange the systems to VAR(1)
mx    = np.zeros(num_vars); mx[:n] = B[:,0]            # coefficient on constant
A     = B[:,1:]                                        # coefficient on lags
G     = np.block([[A],[np.eye(L * n - n), np.zeros((L * n - n, n))]])
G     = G[noncinds]; G = G[:,noncinds]                 # This is the matrix A from the paper
V     = np.linalg.cholesky(S)                          # Cholusky decomposition of variance-cov
H     = np.block([[V],[np.zeros((L * n - n - 1, n))]]) # Corresponds to the matrix B from the paper
uc    = np.zeros(np.sum(lags)); uc[0] = 1              # Used to select consumption components of vectors
BB    = H@H.T
Sigma = la.solve_discrete_lyapunov(G,BB)               # Written as Sigma in the paper
mu0   = la.solve(np.eye(len(G))-G, mx)                 # Written as mu in the paper

####################################################
#  Construct MLE estimates of relevant parameters  #
####################################################

# These calculations follow appendix B.1 from the paper
print("\nMLE estimates:")
I = la.inv(np.eye(len(G))-G)
alphac = (uc @ I @ mx) * 100

print("\t{}_c: \t{}".format(chr(945), round(alphac, 3)))

coeff = (uc @ G @ I @ G @ Sigma @ I.T @ G.T @ uc.T) /\
                (uc @ G @ I @ Sigma @ I.T @ G.T @ uc.T)
beta = 1 - coeff
print("\t{}_z: \t{}".format(chr(946), round(beta, 3)))
matrixx = np.array([[uc @ BB @ uc.T, uc @ BB @ I.T @ G.T @ uc.T],
           [uc @ G @ I @ BB @ uc.T,  uc @ G @ I @ BB @ I.T @ G.T @ uc.T]])
matrixx = matrixx / 0.0001

sigc1 = np.sqrt(matrixx[0,0])
sigc  = np.array([sigc1, 0])
print("\t{}_c: \t{}".format(chr(963), sigc))
sigz1 = matrixx[1,0] * beta / sigc1
sigz2 = np.sqrt(matrixx[1,1] * beta**2 - sigz1**2)
sigz = np.array([sigz1, sigz2])
print("\t{}_z: \t{}".format(chr(963), sigz))

print("\nDrawing parameters for Monte Carlo:")

T   = len(y_lagged[0])

# Create a triangular system following Zha (1999)
x = []
x.append(X_lagged)
x.append(np.vstack((x[0], y_lagged[0])))
x.append(np.vstack((x[1], y_lagged[1])))

# Coefficients from regression
b_hat0 = []
# Part of precision matrix for coefficient draws following Zha
Lam = []
# Part of gamma distribution of scaling coefficient \zeta in Zha
dt = []

for k in range(n):
    b_hat0.append(la.solve(x[k]@x[k].T, x[k]@y_lagged[k]))
    Lam.append(x[k] @ x[k].T)
    et = y_lagged[k] - x[k].T@b_hat0[k] # Residuals
    dt.append(et @ et)

iters = 1000000 # Recommended iterations: 1,000,000

# Get the number of cores available for parallelization
# NOTE: Since the process is being run on all cores, runtime is influenced by
# having other software running on the computer
cpus = os.cpu_count()

# Each iteration must have the random number generator reseeded. This number is
# used as the base for that processor-level seed
current_seed = np.random.get_state()[1][0]

# Used as a matrix invertibility check later on. Make smaller to require better-
# conditioned covariance matrices for the multivariate normal distribution
cond_tol = 1e9

# Gets the locations of the first coefficient associated with each variable
cn = [0] +  np.cumsum(lags).astype(np.int).tolist()

def gen_results(i):
    """
    This function follows Zha to redraw coefficients from the regression and
    re-estimate the model parameters using those coefficients. Each of these
    draws has the ability to be weighted in relative importance by the marginal
    likelihood of X0 given the drawn VAR coefficients. Note that this function
    makes calls to np.block and la.solve_discrete_lyapunov, which as of yet are
    not compatible with numba's jit compiler.

    Input:
    i (int):    Keeps track of the iteration number of the specific call. Used
                    for random number generator seeding
    """
    if current_process().pid % cpus == 0:
        pbar.update(cpus)  # Track progress on one core only

    # Get coefficient draws
    [G, BB, mx] = MLEVARsim(n, T, b_hat0, Lam, dt, lags, cn, i + current_seed, noncinds)
    # Check if the matrix G is explosive; if so, discard the draw. Otherwise, proceed.
    if np.all(np.abs(la.eigvals(G)) <= 1):
        Sigma   = la.solve_discrete_lyapunov(G,BB) # Written as Sigma_j in the paper
        # Check Sigma_j for invertibility conditions
        if np.linalg.cond(Sigma) <= cond_tol:
            return process_VAR(G, Sigma, num_vars, uc, mx, BB, X0)
    return 0, 0, 0, 0, 0, 0, False

# Make a call to each of the jit functions MLEVARsim and process_VAR to compile them
G, BB, mx = MLEVARsim(n, T, b_hat0, Lam, dt, lags, cn, 0, noncinds)
Sigma       = la.solve_discrete_lyapunov(G,BB)                # Written as Sigma in the paper
process_VAR(G, Sigma, num_vars, uc, mx, BB, X0)

start = time.time()
# Create a progress tracker
pbar = tqdm(total = iters)
# Create a parallel pool
pool = Pool(cpus)
# Run the pool, saving the results
res = pool.map(gen_results, range(iters))
pbar.close()
pool.close()
end = time.time()
# Unpack the results from the parallel processes
res = list(zip(*res))
weights, ac, b, sigc1s, sigz1s, sigz2s, valid_runs = [np.array(a) for a in res]

# Discard invalid runs (explosive eigenvalues)
ac = ac[valid_runs]
b = b[valid_runs]
sigc1s = sigc1s[valid_runs]
sigz1s = sigz1s[valid_runs]
sigz2s = sigz2s[valid_runs]
weights = weights[valid_runs]
weights = weights / np.sum(weights)

# Announce that estimation is complete and display useful stats and results
try:
    os.system('say "Estimation complete"')
except:
    pass

print("Finished in {} seconds. {}% of the draws had explosive systems and were discarded.".format(round(end-start,2),round((iters - len(ac)) / iters * 100, 2)))

print("\nUnweighted percentiles:")
acDist = np.array([np.percentile(ac, 10), np.percentile(ac, 50), np.percentile(ac, 90)])
print("\t{}_c:\t{}".format(chr(945),acDist))
betaDist = np.array([np.percentile(b, 10), np.percentile(b, 50), np.percentile(b, 90)])
print("\t{}_z:\t{}".format(chr(946),betaDist))
sigc1Dist = np.array([np.percentile(sigc1s, 10), np.percentile(sigc1s, 50), \
             np.percentile(sigc1s, 90)])
print("\t{}_c^1:\t{}".format(chr(963),sigc1Dist))
sigz1Dist = np.array([np.percentile(sigz1s, 10), np.percentile(sigz1s, 50), \
             np.percentile(sigz1s, 90)])
print("\t{}_z^1:\t{}".format(chr(963),sigz1Dist))
sigz2Dist = np.array([np.percentile(sigz2s, 10), np.percentile(sigz2s, 50), \
             np.percentile(sigz2s, 90)])
print("\t{}_z^2:\t{}".format(chr(963),sigz2Dist))

print("Weighted percentiles:")
acDist = np.array([wprctile(ac, weights, 10), wprctile(ac, weights, 50), \
          wprctile(ac, weights, 90)])
print("\t{}_c:\t{}".format(chr(945),acDist))
betaDist = np.array([wprctile(b, weights, 10), wprctile(b, weights, 50), \
            wprctile(b, weights, 90)])
print("\t{}_z:\t{}".format(chr(946),betaDist))
sigc1Dist = np.array([wprctile(sigc1s, weights, 10), wprctile(sigc1s, weights, 50), \
             wprctile(sigc1s, weights, 90)])
print("\t{}_c^1:\t{}".format(chr(963),sigc1Dist))
sigz1Dist = np.array([wprctile(sigz1s, weights, 10), wprctile(sigz1s, weights, 50), \
             wprctile(sigz1s, weights, 90)])
print("\t{}_z^1:\t{}".format(chr(963),sigz1Dist))
sigz2Dist = np.array([wprctile(sigz2s, weights, 10), wprctile(sigz2s, weights, 50), \
             wprctile(sigz2s, weights, 90)])
print("\t{}_z^2:\t{}".format(chr(963),sigz2Dist))

#######################
#   Generate graphs   #
#######################

# Define the current variable (beta_z here)
xint = b
# We will discard some outliers
inds = np.logical_and(xint > np.percentile(xint, 1), xint < np.percentile(xint, 99))
# Generate histogram bins
bins = np.linspace(np.percentile(xint, 1), np.percentile(xint, 99), 200)
# Create an histogram where each draw is weighted equally
plt.hist(xint[inds], bins = bins, density = True, \
         alpha = .5, label='Unweighted')
# Add a histogram where each draw is weighted by the marginal likelihood of X0
plt.hist(xint[inds], weights = weights[inds], density = True, bins = bins, \
         alpha = .5, label='Weighted')
plt.legend()
plt.title(r"$\beta_z$")
# Save the figure
plt.savefig("beta_z.png")
plt.clf()

xint = ac
inds = np.logical_and(xint > np.percentile(xint, 1), xint < np.percentile(xint, 99))
bins = np.linspace(np.percentile(xint, 1), np.percentile(xint, 99), 200)
plt.hist(xint[inds], bins = bins, density = True, \
         alpha = .5, label='Unweighted')
plt.hist(xint[inds], weights = weights[inds], density = True, bins = bins, \
         alpha = .5, label='Weighted')
plt.legend()
plt.title(r"$\alpha_c$")
plt.savefig("alpha_c.png")
plt.clf()

xint = sigc1s
inds = np.logical_and(xint > np.percentile(xint, 1), xint < np.percentile(xint, 99))
bins = np.linspace(np.percentile(xint, 1), np.percentile(xint, 99), 200)
plt.hist(xint[inds], bins = bins, density = True, \
         alpha = .5, label='Unweighted')
plt.hist(xint[inds], weights = weights[inds], density = True, bins = bins, \
         alpha = .5, label='Weighted')
plt.legend()
plt.title(r"$\sigma_c^1$")
plt.savefig("sigma_c^1.png")
plt.clf()

xint = sigz1s
inds = np.logical_and(xint > np.percentile(xint, 1), xint < np.percentile(xint, 99))
bins = np.linspace(np.percentile(xint, 1), np.percentile(xint, 99), 200)
plt.hist(xint[inds], bins = bins, density = True, \
         alpha = .5, label='Unweighted')
plt.hist(xint[inds], weights = weights[inds], density = True, bins = bins, \
         alpha = .5, label='Weighted')
plt.legend()
plt.title(r"$\sigma_z^1$")
plt.savefig("sigma_z^1.png")
plt.clf()

xint = sigz2s
inds = np.logical_and(xint > np.percentile(xint, 1), xint < np.percentile(xint, 99))
bins = np.linspace(np.percentile(xint, 1), np.percentile(xint, 99), 200)
plt.hist(xint[inds], bins = bins, density = True, \
         alpha = .5, label='Unweighted')
plt.hist(xint[inds], weights = weights[inds], density = True, bins = bins, \
         alpha = .5, label='Weighted')
plt.legend()
plt.title(r"$\sigma_z^2$")
plt.savefig("sigma_z^2.png")
plt.clf()
