import numpy as np
import scipy.linalg as la
from numba import jit, njit
from scipy.stats import multivariate_normal

def MLEVAR(YY,L):
    """
    This function estimates an MLE system using the equation-by-equation approach
    described by Hamilton (1994) p310-312. This function has not been optimized
    for jit compilation since it is only called one time.

    Inputs:
    YY:     A list of numpy arrays of the relevant time series. They may vary in
                length provided that L meets the criterion specified below
    L:      A list of the same length as YY that specifies the lags for each
                variable. It must be specified such that len(YY[i]) - L[i] is the
                same for all variables.

    Returns:
    B:      A numpy array containing the coefficients of regression.
    S:      A numpy array containing the variance of error terms from regression.
    y:      A list of numpy arrays containing the y data appropriately lagged
                for the VAR.
    x:      A numpy array of the x data used in each of the equation-by-equation
                regressions.
    """
    K   = len(YY) # number of blocks in the data
    y   = []
    # X0 will contain the appropriately lagged X data, without the vector of 1s
    X0  = np.empty((np.sum(L), YY[0].shape[0] - L[0]))
    # counter will be used to assign data to the correct rows of X0
    counter = 0
    for k in range(K):
        # Append the appropriately lagged y data to the list y
        y.append(YY[k][L[k]:])
        for i in range(L[k]):
            if i == 0:
                X0[counter] = YY[k][L[k] - i - 1:-1]
            else:
                X0[counter] = YY[k][L[k] - i - 1:-i - 1]
            counter += 1

    # The length of the time period
    T   = len(y[0])
    # et will catch the error vectors
    et = []
    # b_hot0 will contain the regression coefficients
    b_hat0 = []
    # Add the column of ones to the x data
    x  = np.vstack((np.ones(T), X0)).T

    # Run the regressions for each equation
    for k in range(K):
        b_hat0.append(la.solve(x.T@x, x.T@y[k].T))   # n(k) vectors
        et.append(y[k]-x@b_hat0[k])

    et = np.array(et)
    b_hat0 = np.array(b_hat0)
    S   = et@et.T / T # variance of error

    # Rearrange the coefficients
    B0 = b_hat0[:,0, np.newaxis]
    B1 = b_hat0[:,1:]
    cn = [0] +  np.cumsum(L).astype(np.int).tolist() # Gets the locations of the first coefficient associated with each variable
    LL = max(L)
    B = np.empty((K, LL*K))

    # Create equally sized blocks of coefficients for each variable, adding zeros where the lag is smaller than the max lag
    for k in range(K):
        if L[k] < LL:
            temp = np.array(B1[:,cn[k]:cn[k+1]])
            var_coeffs = np.hstack((temp, np.zeros((K, LL-L[k]))))
        else:
            var_coeffs = np.array(B1[:,cn[k]:cn[k+1]])
        B[:,LL*k:LL*(k+1)] = var_coeffs
    inds = [i + LL*j for i in range(LL) for j in range(K)]
    B = B[:,inds]
    B = np.hstack((B0, B))

    return B, S, y, x.T

@jit(nopython=True)
def mvn(mu, sigma):
    """Generate a sample from multivarate normal with mean mu and covariance sigma."""

    A = np.linalg.cholesky(sigma)
    p = len(mu)
    z = np.random.randn(p)
    zs = mu + A@z
    return zs

@jit # This function cannot be used with nopython jit since it creates numpy arrays
def MLEVARsim(K, T, b_hat0, Lam, dt, L, cn, s, noncinds):
    """
    This function takes the results of the uncorrelated VAR described by Zha
    (1999). It then draws new coefficients from the posterior distribution
    and rearranges them into the corresponding objects from the VAR.

    Inputs:
    K:          An integer representing the number of variables in the VAR.
    T:          An integer representing the number of time periods included in the
                    regressions after lags are taken into consideration.
    b_hat0:     A list of numpy arrays containing the results from the uncorrelated
                    regressions.
    Lam:        A list of the lambda matrix used in the precision matrix calculation
                    for the coefficient draws described in Zha.
    dt:         A list of d_ts used to draw the scaling coefficient zeta.
    L:          A list of the number of lags used for each variable in the VAR.
    cn:         A list of the locations of the first coefficients for each variable.
    s:          An integer use to seed the random number generator.
    noncinds:   A list of indices which will be used to drop the extra lag of the
                    consumption growth variable which this system estimates.

    Returns:
    G:          The transition matrix for the VAR system.
    BB:         The one period covariance matrix given the newly drawn coefficients.
    mx:         The coefficients for the constant on each variable.
    """
    # Create empty lists to store the drawn coefficients
    zeta = []
    b_hat1 = []
    # Seed the random number generator
    np.random.seed(s)
    for k in range(K):
        zeta.append(np.random.gamma(T/2 + 1, 2 / dt[k]))
        cov = np.linalg.inv(zeta[k] * Lam[k])
        b_hat1.append(mvn(b_hat0[k], cov))

    # This lower triangular matrix maps from the uncorrelated regressions to the
    # original VAR system
    A1 = np.array([[0,0,0],
                   [b_hat1[1][-1], 0, 0],
                   [b_hat1[2][-2], b_hat1[2][-1], 0]])

    A2 = np.array([b_hat1[0], b_hat1[1][:-1], b_hat1[2][:-2]])

    # Astar contains the regression coefficients once mapped to the original VAR
    Astar = np.linalg.inv(np.eye(len(A1)) - A1) @ A2

    B1star = np.linalg.inv(np.diag(np.sqrt(np.array(zeta))))

    # Part of the matrix B from the system described in the paper
    B1 = np.linalg.inv(np.eye(len(A1)) - A1) @ B1star

    # Rearrange the coefficients in A
    A0 = np.array([Astar[:, 0]]).T
    A1 = Astar[:, 1:]
    LL = np.max(L)
    A = np.empty((K, LL*K))

    # Create equally sized blocks of coefficients for each variable, adding zeros where the lag is smaller than the max lag

    for k in range(K):
        if L[k] < LL:
            temp = np.array(A1[:,cn[k]:cn[k+1]])
            var_coeffs = np.hstack((temp, np.zeros((K, LL-L[k]))))
        else:
            var_coeffs = np.array(A1[:,cn[k]:cn[k+1]])
        A[:,LL*k:LL*(k+1)] = var_coeffs
    inds = [i + LL*j for i in range(LL) for j in range(K)]
    A = A[:,inds]
    A = np.hstack((A0, A))

    # Rearrange results
    G           = A[:,1:]                                         # coefficient on lagged variables
    G           = np.vstack((G, np.hstack((np.eye(LL * K - K), np.zeros((LL * K - K, K))))))
    G           = G[noncinds]; G = G[:,noncinds]                  # This is the matrix A from the paper

    num_vars = np.sum(L)
    mx      = np.zeros(num_vars)
    mx[:K] = A[:,0]             # coefficient on constant

    B1      = np.vstack((B1,np.zeros((LL * K - K - 1, K)))) # Corresponds to the matrix B from the paper, augmented for VAR
    BB      = B1 @ B1.T

    return G, BB, mx

def wprctile(x, w, p):
    """
    Calculates the percentile of an array of data where each point has a weight
    attached to it.

    Inputs:
    x:          A numpy array containing the x data.
    w:          A numpy array containing the weights corresponding to the x data.
                    The entries of w should sum up to 1.
    p:          A float between 0 and 100 representing the percentile you want to
                    calculate.

    Returns:
    x_value:    The entry of x corresponding to the pth percentile under the
                    weighting implied by w.
    """
    sort_indices = np.argsort(x)
    x = x[sort_indices]
    w = w[sort_indices]
    w = np.cumsum(w)
    prctind = np.argmin(np.abs(w - p / 100.))
    x_value = x[prctind]
    return x_value

@njit
def process_VAR(G, Sigma, num_vars, uc, mx, BB, X0):
    """
    This function calculates the implied model parameters given the VAR system
    generated through the process adapted from Zha (1999) and described in the
    appendix.

    Inputs:
    G:          Matrix governing the evolution of the VAR. Written A in the appendix.
    Sigma:      A numpy array describing the covariance of the multivariate normal
                    distribution underlying X0.
    num_vars:   An integer giving the total number of variables contained in X_t.
    uc:         A numpy array which selects the consumption component of the VAR.
    mx:         The average of X_t implied by the VAR.
    BB:         A numpy array corresponding to BB' described by Appendix B.1.
    X0:         A numpy array containing the date zero observation of X_t.

    Returns:
    weight:     A float providing a weight for the parameter draw.
    ac:         The implied value of alpha_c from the paper.
    bet:        The implied value of beta_x from the paper.
    sigc1:      The implied first entry of sigma_c.
    sigz1:      The implied first entry of sigma_z.
    sigz2:      The implied second entry of sigma_z.
    valid_run:  Denotes that this parameter setting did not have explosive eigenvalues.
    """
    mu0 = np.linalg.solve(np.eye(len(G))-G, mx)           # Written as mu_j in the paper
    # weight = multivariate_normal.pdf(X0, mean=mu0, cov=Sigma)
    weight = (2 * np.pi) ** (-num_vars / 2) * np.linalg.det(Sigma) ** -.5 * \
                np.exp(-.5 * (X0 - mu0).T @ np.linalg.inv(Sigma) @ (X0 - mu0))
    I = np.linalg.inv(np.eye(len(G))-G)
    ac = (uc @ I @ mx) * 100

    bet = 1 - (uc @ G @ I @ G @ Sigma @ I.T @ G.T @ uc.T) /\
                    (uc @ G @ I @ Sigma @ I.T @ G.T @ uc.T)
    matrixx = np.array([[uc @ BB @ uc.T, uc @ BB @ I.T @ G.T @ uc.T],
               [uc @ G @ I @ BB @ uc.T, uc @ G @ I @ BB @ I.T @ G.T @ uc.T]]) / 0.0001
    sigc1 = np.sqrt(matrixx[0,0])
    sigz1 = matrixx[1,0] * bet / sigc1
    sigz2 = np.sqrt(matrixx[1,1] * bet**2 - sigz1**2)
    valid_run = True
    return weight, ac, bet, sigc1, sigz1, sigz2, valid_run
