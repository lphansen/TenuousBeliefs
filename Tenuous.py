import numpy as np
import scipy.stats
from numpy.linalg import norm, det, inv
import sympy
import sys
import os
from IPython.core.display import display, HTML
import plotly.io as pio
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pickle
from scipy.integrate import solve_bvp
from scipy.optimize import fsolve, minimize
from scipy.interpolate import CubicSpline, interp1d
from numpy.linalg import solve, eig
from scipy.io import loadmat
import scipy.sparse
from scipy.sparse.linalg import spsolve
import copy
import datetime


# To-dos:
# 1. drift comparison for arbitrary qus


params = {}
params['q'] = 0.05

params['αk'] = 0.484  #0.386
params['αz'] = 0
params['βk'] = 1
params['βz'] = 0.014
params['σy'] = np.array([[0.477], [0]])
params['σz'] = np.array([[0.011], [0.025]])
params['δ'] = 0.002

params['ρ1'] = 0

params['ρ2'] = params['q'] ** 2 / norm(params['σz']) ** 2
params['z̄'] = params['αz'] / params['βz']
params['σ'] = np.vstack([params['σy'].T, params['σz'].T ]) 
params['a'] = norm(params['σz']) ** 2 /  det(params['σ'] ) ** 2
params['b'] = - np.squeeze(params['σy'].T.dot(params['σz'])) /  det(params['σ'] ) ** 2
params['d'] = norm(params['σy']) ** 2 /  det(params['σ'] ) ** 2

params['zl'] = -2.5
params['zr'] = 2.5
# print(params)
ρ2_default = params['ρ2']

def FeynmanKac(μz, σz, zgrid, fintl, T, Dt):
    Dz = zgrid[1] - zgrid[0]
    Nz = len(zgrid)
    row = []
    col = []
    value = []
    for j in range(Nz):
        if j == 0:
            value.append( -1)
            row.append(j)
            col.append(j)
        elif j == Nz - 1:
            value.append(-1)
            row.append(j)
            col.append(j)
        else:
            value.append( -1 + Dt * (-norm(σz) ** 2 / Dz ** 2))
            row.append(j)
            col.append(j)
            value.append( Dt * (μz[j] / (2*Dz) + 0.5 * norm(σz) ** 2 / (Dz ** 2)))
            row.append(j)
            col.append(j+1)
            value.append( Dt * (-μz[j] / (2*Dz) + 0.5 * norm(σz) ** 2 / (Dz ** 2)))
            row.append(j)
            col.append(j-1)
    A = scipy.sparse.csr_matrix((value, (row, col)))
    a1 = A[1,0]
    a2 = A[-2,-1]
    A = A[1:-1, 1:-1]

    ϕold = fintl
    sol = np.zeros([Nz, int(T/Dt) + 1])
    sol[:,0] = fintl

    for t in range(int(T/Dt)):
        b = - ϕold[1:-1]
        b[0] = b[0] - a1 * ϕold[0]
        b[-1] = b[-1] -a2 * ϕold[-1]
        ϕnew = spsolve(A, b)
        ϕnew = np.hstack([2 * ϕnew[0] - ϕnew[1], ϕnew, 2 * ϕnew[-1] - ϕnew[-2]])
        ϕold = ϕnew
        sol[:,t+1] = ϕnew
    return sol

def InterpQuantile(zgrid, mgrid, z0):
    dim = mgrid.shape
    res = []
    for t in range(dim[1]):
        f = interp1d(zgrid, mgrid[:,t])
        res.append(f(z0))
    return np.array(res)

class SturcturedModel():

    def __init__(self, params, q0s, qᵤₛ, ρ2 = None):
        self.θ = None
        self.αŷ = params['αŷ']
        self.αẑ = params['αẑ']
        self.β̂  = params['β̂']
        self.κ̂  = params['κ̂']
        self.σy = params['σy']
        self.σz = params['σz']
        self.δ = params['δ']
        self.ρ1 = params['ρ1']

        # Parameters that could alter
        self.z̄ = params['z̄']
        self.σ = params['σ']
        self.a = params['a']
        self.b = params['b']
        self.d = params['d']
        self.q0s = q0s
        self.qᵤₛ = qᵤₛ

        if ρ2 is None:
            self.ρ2 = self.q0s ** 2 / norm(params['σz']) ** 2
        else:
            self.ρ2 = ρ2

        # self.zrange = [-2.5, 2.5]
        self.zl = params['zl']
        self.zr = params['zr']
        self.Dz = 0.01
        
        self.x = None # this is the z grid
        self.y = None

        self.s1 = None
        self.s2 = None

        self.status = 0  # 0: Not solved; 1: solved
        self.dvErr = None
        self.qErr = None
        self.dv0 = None
        self.hl = None
        self.v = None
        self.Distorted = None


    def HJBODE(self, z, v, θ):
        v0 = v[0]
        v1 = v[1]
        if isinstance(v1, (int, float, np.float)):
            (min_val, _) = self.mined(z, v1)
    #         print(min_val)
            temp = np.array([0.01, v1]).dot(self.σ).dot(self.σ.T).dot(np.array([[0.01],[v1]]))
    #         print("z: {}; v1: {}; min_val: {}".format(z, v1, min_val))
    #         print("min_val: {}; dot prod: {}; v0: {}".format(min_val, temp, v0))
            return np.vstack((v1,
                            2 / norm(self.σz) ** 2 * (self.δ * v0 - min_val + 1 / (2 * θ) *  temp )))
            
        else:
            min_val = self.mined(z, v1)
            temp = np.zeros(len(v1))
            for i in range(len(v1)):
        #         print(np.array([[0.01],[v1[i]]]))
        #         print(np.array([0.01, v1[i]]))
                temp[i] = np.array([0.01, v1[i]]).dot(self.σ).dot(self.σ.T).dot(np.array([[0.01],[v1[i]]]))

            return np.vstack((v1,
                            2 / norm(self.σz) ** 2 * (self.δ * v0 - min_val + 1 / (2 * θ) *  temp )))

    def mined(self, z, dv):
        
        A = 0.5 * self.a
        C0 = (self.ρ1 + self.ρ2 * (z - self.z̄)) * (self.αẑ - self.κ̂ * (z - self.z̄)) + norm(self.σz) ** 2 / 2 * self.ρ2 - self.q0s ** 2 / 2
        C1 = (self.ρ1 + self.ρ2 * (z - self.z̄))
        C2 = 0.5 * self.d
        D = self.b ** 2 / (2 * A) - 2 * C2
        E = (100 * dv - self.b / (2*A)) ** 2
        
        AA = E * self.b **2 - 4 * A * E * C2 - D ** 2
        BB = 2 * C1 * D - 4 * A * E * C1
        CC = - 4 * A * C0 * E - C1 ** 2
        
        s21 = ( -BB + np.sqrt(BB ** 2 - 4 * AA * CC)) / (2 * AA)
        s22 = ( -BB - np.sqrt(BB ** 2 - 4 * AA * CC)) / (2 * AA)
        
        mined1 = np.squeeze(0.01 * (self.αŷ + self.β̂ * (z-self.z̄) + self.S1(s21, z)) + dv * (self.αẑ - self.κ̂ * (z - self.z̄) + s21))
        mined2 = np.squeeze(0.01 * (self.αŷ + self.β̂ * (z-self.z̄) + self.S1(s22, z)) + dv * (self.αẑ - self.κ̂ * (z - self.z̄) + s22))
        
        if isinstance(mined1, (int, float, np.float)):
            res = min(mined1, mined2)
            return (res, s21 * (mined1 <= mined2) + s22 * (mined1 > mined2))
        else:
            res = np.min(np.vstack([mined1,mined2]), axis = 0)
            return res

    def S1(self, s2, z):
    
        A = 0.5 * self.a
        B = self.b * s2
        C = 0.5 * self.d * s2 ** 2 + (self.ρ1 + self.ρ2 * (z - self.z̄)) * s2 + (self.ρ1 + self.ρ2 * (z - self.z̄)) * (self.αẑ - self.κ̂ * (z - self.z̄)) + norm(self.σz) ** 2 / 2 * self.ρ2 - self.q0s ** 2 / 2
        
        return (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

    def ApproxBound(self):
    
        ν, s1, s2 = sympy.symbols('ν s1 s2')
        f1 = (-self.δ - self.κ̂ + s2) * ν + 0.01 * (self.β̂ + s1)
        f2 = ν * (self.a * s1 + self.b * s2) - 0.01 * (self.b * s1 + self.d * s2 + self.ρ2)
        f3 = 0.5 * (self.a * s1 ** 2 + 2 * self.b * s1 * s2 + self.d * s2 ** 2) + self.ρ2 * (-self.κ̂ + s2 )
        # initialGuess = (np.array([0.2, 0.8]), np.array([-0.1, 0.1]), np.array([0, 0]))
        bounds =  np.array(sympy.solvers.solve((f1, f2, f3), (ν, s1, s2))).astype(float)
        self.dvl = max(bounds[:,0])
        self.dvr = min(bounds[:,0])
    
    def ODEsolver(self, zrange, bdl, bdr, θ):
        def tosolve(z, v):
            return self.HJBODE(z, v, θ)
        
        def bc(ya, yb):
            return np.array([ya[1] - bdl, yb[1] - bdr])
        
        if abs(zrange[0]) >= abs(zrange[1]):
            temp = bdr
        else:
            temp = bdl
            
        x = np.linspace(zrange[0], zrange[1], 10)
        y = np.ones((2,x.size)) * np.array([0, temp])[:,np.newaxis]
        res = solve_bvp(tosolve, bc, x, y)
        return res
        
    def MatchODE(self, θ, dv0guess = None):
        res = {}
        
        def v0Diff(dv0):
    #         print(type(dv0))
    #         if type(dv0) is list or np.ndarray:
    #             dv0 = dv0[0]
            # print('trying dv(0) = {}'.format(dv0))
            negsol = self.ODEsolver([self.zl, 0], self.dvl, dv0, θ)
            # print('For this case, v(0-) = {}'.format(negsol['y'][0,-1]))
            possol = self.ODEsolver([0, self.zr], dv0, self.dvr, θ)
            # print('For this case, v(0+) = {}'.format(possol['y'][0, 0]))

            diff = negsol['y'][0, -1] - possol['y'][0, 0]
            # print('The difference is: {}'.format(diff))
            return diff
        if dv0guess is None:
            dv0_lists = np.linspace(self.dvr + 0.2 * (self.dvl - self.dvr), self.dvl - 0.2 * (self.dvl - self.dvr), 5)
            gridsvalue = []
            for init in dv0_lists:
                gridsvalue.append(v0Diff(init))

            minIdx = np.argmin(abs(np.array(gridsvalue)))
            dv0guess = dv0_lists[minIdx]
            # print('Grids: {}; Values: {}'.format(dv0_lists, gridsvalue))


        dv0 = np.squeeze(fsolve(v0Diff, dv0guess))
    #     return np.squeeze(fsolve(v0Diff, dv0guess))

        # print('-----------------------')
        # print('dv matched at {} with Error {}'.format(dv0, v0Diff(dv0)))
        
        negsol = self.ODEsolver([self.zl, 0], self.dvl, dv0, θ)
        v0 = negsol.y[0,-1]
        v1 = negsol.y[1,-1]
    #     print(type(v1))
        (min_val,_) = self.mined(-1e-6, v1)
        v2 = 2 / norm(self.σz) ** 2 * (self.δ * v0 - min_val + 1 / (2 * θ) *  np.array([0.01, v1]).dot(self.σ).dot(self.σ.T).dot(np.array([[0.01],[v1]])))
        # print("For θ = {}, v(0-) = {}; v'(0-) = {}; v''(0-) = {}".format(θ, v0, v1, v2))
    #     return negsol
        possol = self.ODEsolver([0, self.zr], dv0, self.dvr, θ)
        v0 = possol.y[0, 0]
        v1 = possol.y[1, 0]
        (min_val,_) = self.mined(1e-6, v1)
        v2 = 2 / norm(self.σz) ** 2 * (self.δ * v0 - min_val + 1 / (2 * θ) *  np.array([0.01, v1]).dot(self.σ).dot(self.σ.T).dot(np.array([[0.01],[v1]])))
        # print("For θ = {}, v(0+) = {}; v'(0+) = {}; v''(0+) = {}".format(θ, v0, v1, v2))
        
        x_neg = np.append(np.arange(-2.5, 0, self.Dz), 0)
        negSpline = CubicSpline(negsol.x, negsol.y, axis = 1)
        negSplined = negSpline(x_neg)
        
        x_pos = np.append(np.arange(0, 2.5, self.Dz), 2.5)
        posSpline = CubicSpline(possol.x, possol.y, axis = 1)
        posSplined = posSpline(x_pos)
        
        res['x'] = np.hstack([x_neg, x_pos[1:]])
        res['y'] = np.hstack([negSplined, posSplined[:,1:]])
        res['possol'] = possol
        res['negsol'] = negsol
        res['diff'] = abs(v0Diff(dv0))
        res['dv0'] = dv0
        return res

    def Distortion(self, sol, θ):
        # Calculate R
        Nz = len(sol['x'])
        s2 = np.zeros(Nz)
        s1 = np.zeros(Nz)
        
        for j in range(Nz):
            (_, s2[j]) = self.mined(sol['x'][j], sol['y'][1,j])
            s1[j] = self.S1(s2[j], sol['x'][j])
            
        s = np.vstack([s1,s2])
        r = solve(self.σ, s)
        
        h = -1 / θ * self.σ.T.dot(np.vstack([np.ones([1,Nz]) * 0.01, sol['y'][1,:]])) + r
        
        rh = np.vstack([r, h])
        
        return (rh, s1, s2)

    def RelativeEntropyUS(self, ηᵤ ,ηₛ , zgrid):
        Nz = len(zgrid)
        Q = np.zeros([Nz, Nz])
        for j in range(Nz):
            if j == 0:
                Q[0, 0] = (self.σz.T.dot(ηᵤ[:,j]) + self.αẑ - self.κ̂ * zgrid[j]) / (self.Dz) - 0.5 * norm(self.σz) ** 2 / (self.Dz ** 2)
                Q[0, 1] = -(self.σz.T.dot(ηᵤ[:,j]) + self.αẑ - self.κ̂ * zgrid[j]) / (self.Dz) + norm(self.σz) ** 2 / (self.Dz ** 2)
                Q[0, 2] = -0.5 * norm(self.σz) ** 2 / self.Dz ** 2
                
            elif j == Nz - 1:
                Q[j, j] = -(self.σz.T.dot(ηᵤ[:,j]) + self.αẑ - self.κ̂ * zgrid[j]) / (self.Dz) - 0.5 * norm(self.σz) ** 2 / (self.Dz ** 2)
                Q[j, j - 1] = (self.σz.T.dot(ηᵤ[:,j]) + self.αẑ - self.κ̂ * zgrid[j]) / (self.Dz) + norm(self.σz) ** 2 / (self.Dz ** 2)
                Q[j, j - 2] = -0.5 * norm(self.σz) ** 2 / self.Dz ** 2
                    
            else:
                
                Q[j, j - 1] = (self.σz.T.dot(ηᵤ[:,j]) + self.αẑ - self.κ̂ * zgrid[j]) / (2 * self.Dz) - 0.5 * norm(self.σz) ** 2 / (self.Dz ** 2)
                Q[j, j] = norm(self.σz) ** 2 / self.Dz ** 2
                Q[j, j + 1] = -(self.σz.T.dot(ηᵤ[:,j]) + self.αẑ - self.κ̂ * zgrid[j]) / (2 * self.Dz) - 0.5 * norm(self.σz) ** 2 / (self.Dz ** 2)
                
        tmp = ηᵤ - ηₛ
        rhs = (tmp[0,:] ** 2 + tmp[1,:] ** 2) / 2
        lhs = Q
        lhs[:,zgrid == self.z̄] = 1
        sol = solve(lhs, rhs)
        q = np.sqrt(sol[zgrid == self.z̄] * 2)
        return q

    def CalibratingTheta(self, θ, dv0, gridsearch = False):
        if gridsearch:
            res = self.MatchODE(θ,None)
            if res['diff'] > 1:
                return np.inf
            else:
                (Distorted, _, _) = self.Distortion(res, θ)
                qᵤₛ = self.RelativeEntropyUS(Distorted[2:,:],Distorted[:2, :], res['x'])
                return qᵤₛ - self.qᵤₛ
        else:

            res = self.MatchODE(θ,None)
            self.dvErr = res['diff']
            (Distorted, _, _) = self.Distortion(res, θ)
            qᵤₛ = self.RelativeEntropyUS(Distorted[2:,:],Distorted[:2, :], res['x'])
            self.qErr = qᵤₛ - self.qᵤₛ
            # print(qᵤₛ - self.qᵤₛ)
            return qᵤₛ - self.qᵤₛ

    def solvetheta(self, dv0):
        if self.qᵤₛ == np.inf:
            self.θ =  np.inf
            self.status = 1
        else:
            thetalist = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2]
            values = []
            for theta in thetalist:
                values.append(self.CalibratingTheta(theta, dv0, gridsearch = True))

            
            minIdx = np.argmin(abs(np.array(values)))
            theta0guess = thetalist[minIdx]

            self.θ = np.squeeze(fsolve(self.CalibratingTheta, theta0guess, (dv0, False), maxfev = 20))
            if self.qErr < 1e-2 and self.dvErr < 1e-4:
                self.status = 1
    
    def HL(self, calHL):
        res = self.MatchODE(self.θ, self.dv0)
        (Distorted, s1, s2) = self.Distortion(res, self.θ)

        self.v = res
        self.Distorted = Distorted
        self.s1 = s1
        self.s2 = s2
        
        if calHL:
            ρ = self.ChernoffEntropy(Distorted[2:,:])
            hl = np.log(2) / ρ
        else:
            hl = None
        
        self.hl = hl
         
    def ChernoffEntropy(self, η):
        def Rhos(s):

            Nz = len(self.v['x'])
        
            Q = np.zeros([Nz, Nz])
            for j in range(Nz):
                if j == 0:
                    Q[0, 0] = -s * (1-s) / 2 * norm(η[:,j]) ** 2 - (s * self.σz.T.dot(η[:,j]) + self.αẑ - self.κ̂ * self.v['x'][j]) / (self.Dz) + 0.5 * norm(self.σz) ** 2 / (self.Dz ** 2)
                    Q[0, 1] = (s * self.σz.T.dot(η[:,j]) + self.αẑ - self.κ̂ * self.v['x'][j]) / (self.Dz) - norm(self.σz) ** 2 / (self.Dz ** 2)
                    Q[0, 2] = 0.5 * norm(self.σz) ** 2 / self.Dz ** 2

                elif j == Nz - 1:
                    Q[j, j] = (-s * (1-s) / 2 * norm(η[:,j]) ** 2) + (s * self.σz.T.dot(η[:,j]) + self.αẑ - self.κ̂ * self.v['x'][j]) / (self.Dz) + 0.5 * norm(self.σz) ** 2 / (self.Dz ** 2)
                    Q[j, j - 1] = -(s * self.σz.T.dot(η[:,j]) + self.αẑ - self.κ̂ * self.v['x'][j]) / (self.Dz) - norm(self.σz) ** 2 / (self.Dz ** 2)
                    Q[j, j - 2] = 0.5 * norm(self.σz) ** 2 / self.Dz ** 2

                else:

                    Q[j, j - 1] = -(s * self.σz.T.dot(η[:,j]) + self.αẑ - self.κ̂ * self.v['x'][j]) / (2 * self.Dz) + 0.5 * norm(self.σz) ** 2 / (self.Dz ** 2)
                    Q[j, j] = - s * (1-s) / 2 * norm(η[:,j]) ** 2 - norm(self.σz) ** 2 / self.Dz ** 2
                    Q[j, j + 1] = (s * self.σz.T.dot(η[:,j]) + self.αẑ - self.κ̂ * self.v['x'][j]) / (2 * self.Dz) + 0.5 * norm(self.σz) ** 2 / (self.Dz ** 2)

            D,_ = eig(Q)
            rhos = max(np.real(D))
            return rhos
        
        res = minimize(Rhos, 0.5, bounds = ((0,1),))

        return -res.fun

    def Drift(self):
        drift = self.σ.dot(self.Distorted[2:,:])
        self.drifty = drift[0,:] + self.αŷ + self.β̂ * (self.v['x'] - self.z̄)
        self.driftz = drift[1,:] + self.αẑ - self.κ̂ * (self.v['x'] - self.z̄)
        
        Nz = len(self.v['x'])
        d2v = np.zeros(Nz)
        for j in range(Nz):
            temp = self.HJBODE(self.v['x'][j], self.v['y'][:,j], self.θ)
            d2v[j] = temp[1]
        
        self.v['y'] = np.vstack([self.v['y'][:2,:], d2v])
        
    def ExpectH(self):
        # unpack parameters
        
        T = 1000
        Dt = 0.1
        
        drift = self.σ.dot(self.Distorted[2:,:])
        μz = drift[1,:] + self.αẑ - self.κ̂ * self.v['x']
        
        h1 = self.Distorted[2,:]
        expectH1 = FeynmanKac(μz, self.σz, self.v['x'], h1, T, Dt)
        mean = self.αẑ / self.κ̂
        std = np.sqrt(norm(self.σz) ** 2 / (2 * self.κ̂))
        z10 = scipy.stats.norm.ppf(0.1, mean, std)
        z90 = scipy.stats.norm.ppf(0.9, mean, std)
        z50 = scipy.stats.norm.ppf(0.5, mean, std)
        
        q10 = InterpQuantile(self.v['x'], expectH1, z10)
        q90 = InterpQuantile(self.v['x'], expectH1, z90)
        q50 = InterpQuantile(self.v['x'], expectH1, z50)
        
        self.shock1 = {'q10': -q10 + 0.01 * self.σy[0],
                    'q50': -q50 + 0.01 * self.σy[0],
                    'q90': -q90 + 0.01 * self.σy[0]}
        
        self.h1 = {'q10': -q10,
                    'q50': -q50,
                    'q90': -q90}
        
        h2 = self.Distorted[3,:]
        expectH2 = FeynmanKac(μz, self.σz, self.v['x'], h2, T, Dt)
        z10 = scipy.stats.norm.ppf(0.1, mean, std)
        z90 = scipy.stats.norm.ppf(0.9, mean, std)
        z50 = scipy.stats.norm.ppf(0.5, mean, std)
        
        q10 = InterpQuantile(self.v['x'], expectH2, z10)
        q90 = InterpQuantile(self.v['x'], expectH2, z90)
        q50 = InterpQuantile(self.v['x'], expectH2, z50)
        
        self.shock2 = {'q10': -q10 + 0.01 * self.σy[1],
                    'q50': -q50 + 0.01 * self.σy[1],
                    'q90': -q90 + 0.01 * self.σy[1]}
        
        self.h2 = {'q10': -q10,
                    'q50': -q50,
                    'q90': -q90}
        
        r1 = self.Distorted[0,:]
        expectR1 = FeynmanKac(μz, self.σz, self.v['x'], r1, T, Dt)
        z10 = scipy.stats.norm.ppf(0.1, mean, std)
        z90 = scipy.stats.norm.ppf(0.9, mean, std)
        z50 = scipy.stats.norm.ppf(0.5, mean, std)
        
        q10 = InterpQuantile(self.v['x'], expectR1, z10)
        q90 = InterpQuantile(self.v['x'], expectR1, z90)
        q50 = InterpQuantile(self.v['x'], expectR1, z50)
        
        self.ambiguity1 = {'q10': -q10,
                    'q50': -q50,
                    'q90': -q90}
        
        r2 = self.Distorted[1,:]
        expectR2 = FeynmanKac(μz, self.σz, self.v['x'], r2, T, Dt)
        z10 = scipy.stats.norm.ppf(0.1, mean, std)
        z90 = scipy.stats.norm.ppf(0.9, mean, std)
        z50 = scipy.stats.norm.ppf(0.5, mean, std)
        
        q10 = InterpQuantile(self.v['x'], expectR2, z10)
        q90 = InterpQuantile(self.v['x'], expectR2, z90)
        q50 = InterpQuantile(self.v['x'], expectR2, z50)
        
        self.ambiguity2 = {'q10': -q10,
                    'q50': -q50,
                    'q90': -q90}
        
        self.misspec1 = {'q10': -self.ambiguity1['q10'] + self.h1['q10'],
                    'q50': -self.ambiguity1['q50'] + self.h1['q50'],
                    'q90': -self.ambiguity1['q90'] + self.h1['q90']}
        
        self.misspec2 = {'q10': -self.ambiguity2['q10'] + self.h2['q10'],
                    'q50': -self.ambiguity2['q50'] + self.h2['q50'],
                    'q90': -self.ambiguity2['q90'] + self.h2['q90']}

class TenuousModel():

    def __init__(self, param = params, q0s = [0.05, 0.1], qus = [0.1, 0.2], ρs = [0.5, 1], load = True):
        self.params = {}
        self.params['αŷ'] = params['αk']
        self.params['αẑ'] = params['αz']
        self.params['β̂'] = params['βk']
        self.params['κ̂'] = params['βz']
        self.params['σy'] = params['σy']
        self.params['σz'] = params['σz']
        self.params['δ'] = params['δ']

        self.params['ρ1'] = params['ρ1']
        # self.ρ2 = params['ρ2']
        self.params['z̄'] = params['z̄']
        self.params['σ'] = params['σ']
        self.params['a'] = params['a']
        self.params['b'] = params['b']
        self.params['d'] = params['d']

        self.params['zrange'] = [-2.5, 2.5]
        self.params['Dz'] = 0.01
        self.params['zr'] = params['zr']
        self.params['zl'] = params['zl']
        
        if not isinstance(q0s, list):
            if isinstance(q0s, (int, float, np.float)):
                q0s = [q0s]
            else:
                q0s = q0s.tolist()
        if np.inf in qus:
            pass
        else:
            qus.append(np.inf)

        if not isinstance(qus, list):
            if isinstance(qus, (int, float, np.float)):
                qus = [qus]
            else:
                qus = qus.tolist()

        if not isinstance(ρs, list):
            if isinstance(ρs, (int, float, np.float)):
                ρs = [ρs]
            else:
                ρs = ρs.tolist()
        
        self.q0s_list = sorted(q0s)
        self.qus_list = sorted(qus)
        self.ρ_list = sorted(ρs)
        self.models = {}
        
        # if load == True:
        #     if not os.path.isfile('tenuousmodel.pickle'):
        #         for q0s in self.q0s_list:
        #             for qus in self.qus_list:
        #                 ρ_restricted = q0s ** 2 / norm(self.params['σz']) ** 2
        #                 for ρ in self.ρ_list:
        #                     print("q0s = {}; qus = {}; rho2 = {};".format(q0s, qus, ρ * ρ_restricted))
        #                     if ρ == 1:
        #                         self.models[q0s, qus, ρ] = SturcturedModel(self.params, q0s, qus)
        #                         self.models[q0s, qus, ρ].ApproxBound()
        #                         self.models[q0s, qus, ρ].solvetheta(None)
        #                         self.models[q0s, qus, ρ].HL(calHL = True)
        #                         self.models[q0s, qus, ρ].Drift()
        #                         self.models[q0s, qus, ρ].ExpectH()

        #                     else:
        #                         # if (q0s % 0.05 == 0) and (qus % 0.1 == 0):
        #                         self.models[q0s, qus, ρ] = SturcturedModel(self.params, q0s, qus, ρ * ρ_restricted)
        #                         self.models[q0s, qus, ρ].ApproxBound()
        #                         self.models[q0s, qus, ρ].solvetheta(None)
        #                         self.models[q0s, qus, ρ].HL(calHL = True)
        #                         self.models[q0s, qus, ρ].Drift()
        #                         self.models[q0s, qus, ρ].ExpectH()

        #                         # else:
        #                         #     print('pass')
        #                         #     pass
                                                                    

            
        #         with open('tenuousmodel.pickle', "wb") as file_:
        #             pickle.dump(self.models, file_, -1)

        #     else:
        #         self.models = pickle.load(open('tenuousmodel.pickle', "rb", -1))
        # else:
        #     for q0s in self.q0s_list:
        #         for qus in self.qus_list:
        #             ρ_restricted = q0s ** 2 / norm(self.params['σz']) ** 2
        #             for ρ in self.ρ_list:
        #                 print("q0s = {}; qus = {}; rho2 = {};".format(q0s, qus, ρ * ρ_restricted))
        #                 if ρ == 1:
        #                     self.models[q0s, qus, ρ] = SturcturedModel(self.params, q0s, qus)
        #                     self.models[q0s, qus, ρ].ApproxBound()
        #                     self.models[q0s, qus, ρ].solvetheta(None)
        #                     self.models[q0s, qus, ρ].HL(calHL = True)
        #                     self.models[q0s, qus, ρ].Drift()
        #                     self.models[q0s, qus, ρ].ExpectH()

        #                 self.models[q0s, qus, ρ] = SturcturedModel(self.params, q0s, qus, ρ * ρ_restricted)
        #                 self.models[q0s, qus, ρ].ApproxBound()
        #                 self.models[q0s, qus, ρ].solvetheta(None)
        #                 self.models[q0s, qus, ρ].HL(calHL = True)
        #                 self.models[q0s, qus, ρ].Drift()
        #                 self.models[q0s, qus, ρ].ExpectH()
      
    def solve(self):
        for q0s in self.q0s_list:
            for qus in self.qus_list:
                ρ_restricted = q0s ** 2 / norm(self.params['σz']) ** 2
                for ρ in self.ρ_list:
                    print("q0s = {}; qus = {}; rho2 = {};".format(q0s, qus, ρ * ρ_restricted))
                    if ρ == 1:
                        self.models[q0s, qus, ρ] = SturcturedModel(self.params, q0s, qus)
                        self.models[q0s, qus, ρ].ApproxBound()
                        self.models[q0s, qus, ρ].solvetheta(None)
                        self.models[q0s, qus, ρ].HL(calHL = True)
                        self.models[q0s, qus, ρ].Drift()
                        self.models[q0s, qus, ρ].ExpectH()

                    self.models[q0s, qus, ρ] = SturcturedModel(self.params, q0s, qus, ρ * ρ_restricted)
                    self.models[q0s, qus, ρ].ApproxBound()
                    self.models[q0s, qus, ρ].solvetheta(None)
                    self.models[q0s, qus, ρ].HL(calHL = True)
                    self.models[q0s, qus, ρ].Drift()
                    self.models[q0s, qus, ρ].ExpectH()

    def driftplot(self):
        fig = go.Figure()
        q0 = self.q0s_list[0]
        rho = self.ρ_list[0]
        qu = self.qus_list[0]
        x = self.models[q0, qu, rho].v['x']
        fig.add_trace(
            go.Scatter(x = x - self.params['z̄'], y = self.models[q0, qu, rho].driftz, 
                name = 'User Setting', legendgroup = 'User Setting', line = dict(color = '#1f77b4', dash = 'solid', width = 3), showlegend = True)) 
        fig.add_trace(
            go.Scatter(x = x - self.params['z̄'], y = self.models[q0, np.inf, rho].driftz, 
                name = 'Worst Case Scenario', legendgroup = 'Worst Case Scenario', line = dict(color = 'red', dash = 'dot', width = 3), showlegend = True)) 
        fig.add_trace(
            go.Scatter(x = x - self.params['z̄'], y = self.params['αẑ'] - self.params['κ̂']* (x - self.params['z̄']), 
                name = 'Baseline Model', legendgroup = 'Baseline Model', line = dict(color = 'black', dash = 'solid', width = 3), showlegend = True))
        fig.update_layout(title = "Growth Rate Drift", titlefont = dict(size = 20))

        fig.update_xaxes(title=go.layout.xaxis.Title(
                                    text="z", font=dict(size=16)), showgrid = False)
        fig.update_yaxes(title=go.layout.yaxis.Title(
                                        text="μz", font=dict(size=16)), showgrid = False)
            
            
        fig.update_xaxes(range = [-0.5, 0.5])
        fig.update_yaxes(range = [-0.025, 0.01])
        fig.update_xaxes(range = [-0.5, 0.5])
        fig.update_yaxes(range = [-0.025, 0.01])
        fig.show()

    def shockplot(self):
        x = np.arange(0, 1000.1 ,0.1)
        q0 = self.q0s_list[0]
        rho = self.ρ_list[0]
        qu = self.qus_list[0]
        fig = make_subplots(rows = 2, cols = 3, print_grid = False, vertical_spacing = 0.08,
                    subplot_titles = (('first shock', 'ambiguity price, first shock', 'misspecification price, first shock',
                                    'second shock', 'ambiguity price, second shock', 'misspecification price, second shock')))
        model = self.models[q0,qu,rho]
        for i,s in enumerate(['shock1', 'ambiguity1', 'misspec1', 'shock2', 'ambiguity2', 'misspec2']):            
            fig.add_trace(go.Scatter(x = x, y = getattr(model, s)['q10'], 
                line = dict(color = 'red', dash = 'dot', width = 3), showlegend = False, legendgroup='.1 decile', name = '.1 decile',
                visible = True), col = i % 3 + 1, row = int((i+3) / 3))
                        
            fig.add_trace(go.Scatter(x = x, y = getattr(model, s)['q50'], 
                line = dict(color = 'Black', dash = 'solid', width = 3), showlegend = False, legendgroup='median', name='median',
                visible = True), col = i % 3 + 1, row = int((i+3) / 3))
            fig.add_trace(go.Scatter(x = x, y = getattr(model, s)['q90'], 
                line = dict(color = '#1f77b4', dash = 'dash', width = 3), showlegend = False, legendgroup='.9 decile', name='.9 decile',
                visible = True), col = i % 3 + 1, row = int((i+3) / 3))

            fig.update_layout(title = "Shock Price Elasticities", titlefont = dict(size = 20), height = 700)

        for i in range(6):
                
            fig['layout']['yaxis{}'.format(i+1)].update(showgrid = False)
            fig['layout']['xaxis{}'.format(i+1)].update(showgrid = False)
        
        for i in range(3,6):
            fig['layout']['xaxis{}'.format(i+1)].update(title=go.layout.xaxis.Title(
                                        text="Horizon(quarters)", font=dict(size=16)), showgrid = False)
                
            
        for i in range(3):
            for j in range(3):
                fig.update_xaxes(range = [0, 40], row = i+1, col = j+1)
                fig.update_yaxes(range = [0, 0.32], row = i+1, col = j+1)
        fig.update_layout(height = 700)
        fig.update_layout(titlefont = dict(size = 20))

        figw = go.FigureWidget(fig)
        display(figw)


    def DriftComparison(self, q0s = [0.05, 0.1], ρs = 1, qus = [0.1, 0.2, np.Inf]): # 
        if type(q0s) == list:  # plot against q
            titles = [r"$\sf q_{{s,0}}: {}$".format(q) for q in q0s]
            fig = make_subplots(rows = 1, cols = len(q0s), print_grid = False, subplot_titles = titles)
            x = self.models[q0s[0], np.inf, 1].v['x']

            for i, q0 in enumerate(q0s):
                if i == 0:
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.2, ρs].driftz, 
                                                name = 'qᵤₛ = .2', legendgroup = 'qᵤₛ = .2', line = dict(color = 'green', dash = 'dashdot', width = 3), showlegend = True),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.1, ρs].driftz, 
                                            name = 'qᵤₛ = .1',legendgroup = 'qᵤₛ = .1', line = dict(color = '#1f77b4', dash = 'solid', width = 3), showlegend = True),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, np.inf, ρs].driftz, 
                                                name = 'Worst Case Scenario', legendgroup = 'Worst Case Scenario', line = dict(color = 'red', dash = 'dot', width = 3), showlegend = True),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.params['αẑ'] - self.params['κ̂']* (x - self.params['z̄']), 
                                                name = 'Baseline Model', legendgroup = 'Baseline Model', line = dict(color = 'black', dash = 'solid', width = 3), showlegend = True),
                            row = 1, col = i +1)
                else:
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.2, ρs].driftz, 
                                                name = 'qᵤₛ = .2', legendgroup = 'qᵤₛ = .2', line = dict(color = 'green', dash = 'dashdot', width = 3), showlegend = False),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.1, ρs].driftz, 
                                            name = 'qᵤₛ = .1',legendgroup = 'qᵤₛ = .1', line = dict(color = '#1f77b4', dash = 'solid', width = 3), showlegend = False),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, np.inf, ρs].driftz, 
                                                name = 'Worst Case Scenario', legendgroup = 'Worst Case Scenario', line = dict(color = 'red', dash = 'dot', width = 3), showlegend = False),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.params['αẑ'] - self.params['κ̂']* (x - self.params['z̄']), 
                                                name = 'Baseline Model', legendgroup = 'Baseline Model', line = dict(color = 'black', dash = 'solid', width = 3), showlegend = False),
                            row = 1, col = i +1)
            
            fig.update_layout(title = r"$\text{Growth rate drift comparisions with different } \sf q_{{0,s}}$", titlefont = dict(size = 20))

        else: # plot against rho

            rho = self.models[q0s, 0.1, 1].ρ2
            titles = [r"$\text{{Relaxed }}\rho_2 = {{\frac{{\sf q_{{s,0}}^2}}{{2|\sigma^2|}}}}$ = {:.2f}".format(rho * ρs[0]),  
                        r"$\text{{Restricted }}\rho_2 = {{\frac{{\sf q_{{s,0}}^2}}{{|\sigma^2|}}}}$ = {:.2f}".format(rho * ρs[1])]
            fig = make_subplots(rows = 1, cols = len(ρs), print_grid = False, subplot_titles = titles)
            q0 = q0s
            x = self.models[q0, np.inf, 1].v['x']
            for i, rs in enumerate(ρs):
                
                if i == 0:
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.2, rs].driftz, 
                                                name = 'qᵤₛ = .2', legendgroup = 'qᵤₛ = .2', line = dict(color = 'green', dash = 'dashdot', width = 3), showlegend = True),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.1, rs].driftz, 
                                            name = 'qᵤₛ = .1',legendgroup = 'qᵤₛ = .1', line = dict(color = '#1f77b4', dash = 'solid', width = 3), showlegend = True),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, np.inf, rs].driftz, 
                                                name = 'Worst Case Scenario', legendgroup = 'Worst Case Scenario', line = dict(color = 'red', dash = 'dot', width = 3), showlegend = True),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.params['αẑ'] - self.params['κ̂']* (x - self.params['z̄']), 
                                                name = 'Baseline Model', legendgroup = 'Baseline Model', line = dict(color = 'black', dash = 'solid', width = 3), showlegend = True),
                            row = 1, col = i +1)
                else:
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.2, rs].driftz, 
                                                name = 'qᵤₛ = .2', legendgroup = 'qᵤₛ = .2', line = dict(color = 'green', dash = 'dashdot', width = 3), showlegend = False),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.1, rs].driftz, 
                                            name = 'qᵤₛ = .1',legendgroup = 'qᵤₛ = .1', line = dict(color = '#1f77b4', dash = 'solid', width = 3), showlegend = False),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, np.inf, rs].driftz, 
                                                name = 'Worst Case Scenario', legendgroup = 'Worst Case Scenario', line = dict(color = 'red', dash = 'dot', width = 3), showlegend = False),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.params['αẑ'] - self.params['κ̂']* (x - self.params['z̄']), 
                                                name = 'Baseline Model', legendgroup = 'Baseline Model', line = dict(color = 'black', dash = 'solid', width = 3), showlegend = False),
                            row = 1, col = i +1)
            fig.update_layout(title = r"$\text{Growth rate drift comparisions betweeen restricted and unrestricted } \rho_2$", titlefont = dict(size = 20))

        for i in range(2):
                
            fig['layout']['yaxis{}'.format(i+1)].update(showgrid = False)
            fig['layout']['xaxis{}'.format(i+1)].update(showgrid = False)
            fig['layout']['xaxis{}'.format(i+1)].update(title=go.layout.xaxis.Title(
                                        text="z", font=dict(size=16)), showgrid = False)
                
        fig['layout']['yaxis1'].update(title=go.layout.yaxis.Title(
                                        text="μz", font=dict(size=16)), showgrid = False)
            
            
        fig.update_xaxes(range = [-0.5, 0.5], row = 1, col = 1)
        fig.update_yaxes(range = [-0.025, 0.01], row = 1, col = 1)
        fig.update_xaxes(range = [-0.5, 0.5], row = 1, col = 2)
        fig.update_yaxes(range = [-0.025, 0.01], row = 1, col = 2)
        fig.show()
    
    def Figure6(self, q0s = [0.05, 0.1], qus = 0.2):
        x = np.arange(0, 1000.1 ,0.1)
        fig = make_subplots(rows = 2, cols = 2, print_grid = False, vertical_spacing = 0.08,
                    subplot_titles = (('first shock with qₛₒ = {:.2f}'.format(q0s[0]), 'first shock with qₛₒ = {:.2f}'.format(q0s[1]),\
                                    'second shock with qₛₒ = {:.2f}'.format(q0s[0]), 'second shock with qₛₒ = {:.2f}'.format(q0s[1]))))
        for i, s in enumerate(['shock1', 'shock2']):
            for j, q in enumerate(q0s):
                model = self.models[q, qus, 1]
                fig.add_trace(go.Scatter(x = x, y = getattr(model, s)['q10'], 
                                line = dict(color = 'red', dash = 'dot', width = 3), showlegend = False, legendgroup='.1 decile', name = '.1 decile'),
                                row = i + 1, col = j + 1) 
                fig.add_trace(go.Scatter(x = x, y = getattr(model, s)['q50'], 
                                line = dict(color = 'Black', dash = 'solid', width = 3), showlegend = False, legendgroup='median', name='median'),
                                row = i + 1, col = j + 1) 
                fig.add_trace(go.Scatter(x = x, y = getattr(model, s)['q90'], 
                                line = dict(color = '#1f77b4', dash = 'dash', width = 3), showlegend = False, legendgroup='.9 decile', name='.9 decile'),
                                row = i + 1, col = j + 1) 
        fig.data[0]['showlegend'] = True
        fig.data[1]['showlegend'] = True
        fig.data[2]['showlegend'] = True
        for i in range(4):
                
            fig['layout']['yaxis{}'.format(i+1)].update(showgrid = False)
            fig['layout']['xaxis{}'.format(i+1)].update(showgrid = False)
        for i in range(2,4):
            fig['layout']['xaxis{}'.format(i+1)].update(title=go.layout.xaxis.Title(
                                        text="Horizon(quarters)", font=dict(size=16)), showgrid = False)

        for i in range(2):
            for j in range(2):
                fig.update_xaxes(range = [0, 40], row = i+1, col = j+1)
                fig.update_yaxes(range = [0, 0.32], row = i+1, col = j+1)
        fig.update_layout(height = 700)
        fig.update_layout(title = r"$\text{Shock price elasticities with different }\sf q_{s,0}$", titlefont = dict(size = 20))
        fig.show()

    def Figure7(self, q0s = 0.1, qus = 0.2):
        x = np.arange(0, 1000.1 ,0.1)
        fig = make_subplots(rows = 2, cols = 2, print_grid = False, vertical_spacing = 0.08,
                    subplot_titles = (('ambiguity price for the first shock', 'misspecification price for the first shock',\
                                    'ambiguity price for the second shock', 'misspecification price for the second shock')))
        for i, s in enumerate(['ambiguity1', 'ambiguity2', 'misspec1', 'misspec2']):
            model = self.models[q0s, qus, 1]
            fig.add_trace(go.Scatter(x = x, y = getattr(model, s)['q10'], 
                            line = dict(color = 'red', dash = 'dot', width = 3), showlegend = False, legendgroup='.1 decile', name = '.1 decile'),
                        row = (i+1) % 2 + 1, col = int((i+2) / 2))
            fig.add_trace(go.Scatter(x = x, y = getattr(model, s)['q50'], 
                            line = dict(color = 'Black', dash = 'solid', width = 3), showlegend = False, legendgroup='median', name='median'),
                        row = (i+1) % 2 + 1, col = int((i+2) / 2))
            fig.add_trace(go.Scatter(x = x, y = getattr(model, s)['q90'], 
                            line = dict(color = '#1f77b4', dash = 'dash', width = 3), showlegend = False, legendgroup='.9 decile', name='.9 decile'),
                        row = (i+1) % 2 + 1, col = int((i+2) / 2))

        fig.data[0]['showlegend'] = True
        fig.data[1]['showlegend'] = True
        fig.data[2]['showlegend'] = True
        for i in range(4):
                
            fig['layout']['yaxis{}'.format(i+1)].update(showgrid = False)
            fig['layout']['xaxis{}'.format(i+1)].update(showgrid = False)

        for i in range(2,4):
            fig['layout']['xaxis{}'.format(i+1)].update(title=go.layout.xaxis.Title(
                                        text="Horizon(quarters)", font=dict(size=16)), showgrid = False)
                
            
        for i in range(2):
            for j in range(2):
                fig.update_xaxes(range = [0, 40], row = i+1, col = j+1)
                fig.update_yaxes(range = [0, 0.32], row = i+1, col = j+1)
        fig.update_layout(height = 700)
        fig.update_layout(title = "Shock price elasticities Decomposition", titlefont = dict(size = 20))
        fig.show()

    def driftIntPlot(self, q0s = None, qus = None):
        fig = go.Figure()
        base = None
        if isinstance(q0s,  (int, float, np.float)): # plot along q0s
            q_list = self.qᵤₛ_list
            for qus in q_list:
                model = self.models[q0s, qus, 1]
                if base is None:
                    base = True
                    fig.add_trace(go.Scatter(x = model.v['x']  - params['z̄'], y = self.params['αẑ'] - self.params['κ̂'] * model.v['x']  - self.params['z̄'],
                        name = 'Baseline Model', 
                        line = dict(color = 'black', dash = 'solid', width = 3), showlegend = True))
                if qus == q_list[int(0.3 * len(q_list))]:
                    fig.add_trace(go.Scatter(x = model.v['x'] - params['z̄'], y = model.driftz, 
                        name = r'$q_{{u,s}} = {}$'.format(qus),
                        line = dict(color = '#1f77b4', dash = 'dash', width = 3), legendgroup = 'Current Line', showlegend = True, visible = True))
                elif qus == np.inf:
                    fig.add_trace(go.Scatter(x = model.v['x'] - params['z̄'], y = model.driftz, 
                        name = 'Worst Case',
                        line = dict(color = '#1f77b4', dash = 'dash', width = 3), legendgroup = 'Current Line', showlegend = True, visible = False)) 
                else:
                    fig.add_trace(go.Scatter(x = model.v['x'] - params['z̄'], y = model.driftz, 
                        name = r'$q_{{u,s}} = {}$'.format(qus),
                        line = dict(color = '#1f77b4', dash = 'dash', width = 3), legendgroup = 'Current Line', showlegend = True, visible = False)) 

            fig.update_layout(title = r"$\text{{Growth rate drift comparisions with }}q_{{0,s}} = {:.2f}$".format(q0s), titlefont = dict(size = 20), height = 700)
            
            steps = []
            for i in range(1, len(q_list) + 1):
                if i == len(q_list):
                    label = 'Worst Case'
                else:
                    label =  '{:.2f}'.format(q_list[i-1])
                step = dict(
                    method = 'restyle',
                    args = ['visible', [False] * len(fig.data)],
                    label = label
                )
                step['args'][1][0] = True
                step['args'][1][i] = True
                
                steps.append(step)
            sliders = [dict(active = int(0.3 * len(q_list)),
                        currentvalue = {"prefix": "qus: "},
                        pad = {"t": len(q_list) },
                        steps = steps, y = -0.1)]

        elif isinstance(qus, (int, float, np.float)):
            q_list = self.q0s_list
            for q0s in q_list:
                model = self.models[q0s, qus, 1]
                if base is None:
                    base = True
                    fig.add_trace(go.Scatter(x = model.v['x']  - params['z̄'], y = self.params['αẑ'] - self.params['κ̂'] * model.v['x']  - self.params['z̄'],
                        name = 'Baseline Model', 
                        line = dict(color = 'black', dash = 'solid', width = 3), showlegend = True))
                if q0s == q_list[int(0.3 * len(q_list))]:
                    fig.add_trace(go.Scatter(x = model.v['x'] - params['z̄'], y = model.driftz, 
                        name = r'$q_{{0,s}} = {}$'.format(q0s),
                        line = dict(color = '#1f77b4', dash = 'dash', width = 3), legendgroup = 'Current Line', showlegend = True, visible = True))
                else:
                    fig.add_trace(go.Scatter(x = model.v['x'] - params['z̄'], y = model.driftz, 
                        name = r'$q_{{0,s}} = {}$'.format(q0s),
                        line = dict(color = '#1f77b4', dash = 'dash', width = 3), legendgroup = 'Current Line', showlegend = True, visible = False)) 
            fig.update_layout(title = r"$\text{{Growth rate drift comparisions with }}q_{{u,s}} = {:.2f}$".format(qus), titlefont = dict(size = 20), height = 600)

            steps = []
            for i in range(1, len(q_list) + 1):
                label =  '{:.2f}'.format(q_list[i-1])
                step = dict(
                    method = 'restyle',
                    args = ['visible', [False] * len(fig.data)],
                    label = label
                )
                step['args'][1][0] = True
                step['args'][1][i] = True
                
                steps.append(step)
            
            sliders = [dict(active = int(0.3 * len(q_list)),
                        currentvalue = {"prefix": "q0s: "},
                        pad = {"t": len(q_list) },
                        steps = steps, y = -0.1)]

        fig.update_layout(xaxis = go.layout.XAxis(title=go.layout.xaxis.Title(
                                            text="z", font=dict(size=16)),
                                                tickfont=dict(size=12), showgrid = False),
                        yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(
                                            text="μz", font=dict(size=16)),
                                                tickfont=dict(size=12), showgrid = False),
                        sliders = sliders
                            )

        fig['layout']['yaxis{}'.format(1)].update(showgrid = False)
        fig['layout']['xaxis{}'.format(1)].update(showgrid = False)
        # fig.update_layout(legend = dict(orientation = 'h', y = 1.1))
        fig.update_xaxes(range = [-0.5, 0.5])
        fig.update_yaxes(range = [-0.025, 0.01])
        figw = go.FigureWidget(fig)
        display(figw)

    def shocksIntPlot(self, q0s = None, qus = None):
        x = np.arange(0, 1000.1 ,0.1)
        fig = make_subplots(rows = 2, cols = 3, print_grid = False, vertical_spacing = 0.08,
                    subplot_titles = (('first shock', 'ambiguity price, first shock', 'misspecification price, first shock',
                                    'second shock', 'ambiguity price, second shock', 'misspecification price, second shock')))
        if isinstance(q0s,  (int, float, np.float)): 
            q_list = self.qᵤₛ_list
            for qus in q_list:
                model = self.models[q0s, qus, 1]
                if qus == q_list[int(0.3 * len(q_list))]:
                    vis = True
                else:
                    vis = False
                for i,s in enumerate(['shock1', 'ambiguity1', 'misspec1', 'shock2', 'ambiguity2', 'misspec2']):
                    # print(i % 3 + 1, int((i+3)/ 3))
                    
                    fig.add_trace(go.Scatter(x = x, y = getattr(model, s)['q10'], 
                        line = dict(color = 'red', dash = 'dot', width = 3), showlegend = False, legendgroup='.1 decile', name = '.1 decile',
                        visible = vis), col = i % 3 + 1, row = int((i+3) / 3))
                                
                    fig.add_trace(go.Scatter(x = x, y = getattr(model, s)['q50'], 
                        line = dict(color = 'Black', dash = 'solid', width = 3), showlegend = False, legendgroup='median', name='median',
                        visible = vis), col = i % 3 + 1, row = int((i+3) / 3))
                    fig.add_trace(go.Scatter(x = x, y = getattr(model, s)['q90'], 
                        line = dict(color = '#1f77b4', dash = 'dash', width = 3), showlegend = False, legendgroup='.9 decile', name='.9 decile',
                        visible = vis), col = i % 3 + 1, row = int((i+3) / 3))

            fig.update_layout(title = r"$\text{{Shock Price Elasticities Decomposition with }}q_{{0,s}} = {:.2f}$".format(q0s), titlefont = dict(size = 20), height = 700)
            steps = []
            for i in range(len(q_list)):
                if i == len(q_list):
                    label = 'Worst Case'
                else:
                    label =  '{:.2f}'.format(q_list[i])
                step = dict(
                    method = 'restyle',
                    args = ['visible', [False] * len(fig.data)],
                    label = label
                )
                for j in range(18):
                    step['args'][1][i * 18 + j] = True
                
                steps.append(step)
            sliders = [dict(active = int(0.3 * len(q_list)),
                        currentvalue = {"prefix": "qus: "},
                        pad = {"t": len(q_list) },
                        steps = steps, y = -0.15)]

        elif isinstance(qus,  (int, float, np.float)): 
            q_list = self.q0s_list
            for q0s in q_list:
                model = self.models[q0s, qus, 1]
                if qus == q_list[int(0.3 * len(q_list))]:
                    vis = True
                else:
                    vis = False
                for i,s in enumerate(['shock1', 'ambiguity1', 'misspec1', 'shock2', 'ambiguity2', 'misspec2']):
                    # print(i % 3 + 1, int((i+3)/ 3))
                    
                    fig.add_trace(go.Scatter(x = x, y = getattr(model, s)['q10'], 
                        line = dict(color = 'red', dash = 'dot', width = 3), showlegend = False, legendgroup='.1 decile', name = '.1 decile',
                        visible = vis), col = i % 3 + 1, row = int((i+3) / 3))
                                
                    fig.add_trace(go.Scatter(x = x, y = getattr(model, s)['q50'], 
                        line = dict(color = 'Black', dash = 'solid', width = 3), showlegend = False, legendgroup='median', name='median',
                        visible = vis), col = i % 3 + 1, row = int((i+3) / 3))
                    fig.add_trace(go.Scatter(x = x, y = getattr(model, s)['q90'], 
                        line = dict(color = '#1f77b4', dash = 'dash', width = 3), showlegend = False, legendgroup='.9 decile', name='.9 decile',
                        visible = vis), col = i % 3 + 1, row = int((i+3) / 3))

            fig.update_layout(title = r"$\text{{Shock Price Elasticities Decomposition with }}q_{{u,s}} = {:.2f}$".format(qus), titlefont = dict(size = 20), height = 700)
            steps = []
            for i in range(len(q_list)):
                label =  '{:.2f}'.format(q_list[i])
                step = dict(
                    method = 'restyle',
                    args = ['visible', [False] * len(fig.data)],
                    label = label
                )
                for j in range(18):
                    step['args'][1][i * 18 + j] = True
                
                steps.append(step)
            sliders = [dict(active = int(0.3 * len(q_list)),
                        currentvalue = {"prefix": "q0s: "},
                        pad = {"t": len(q_list) },
                        steps = steps, y = -0.15)]

        for i in range(6):
                
            fig['layout']['yaxis{}'.format(i+1)].update(showgrid = False)
            fig['layout']['xaxis{}'.format(i+1)].update(showgrid = False)
        
        for i in range(3,6):
            fig['layout']['xaxis{}'.format(i+1)].update(title=go.layout.xaxis.Title(
                                        text="Horizon(quarters)", font=dict(size=16)), showgrid = False)
                
            
        for i in range(3):
            for j in range(3):
                fig.update_xaxes(range = [0, 40], row = i+1, col = j+1)
                fig.update_yaxes(range = [0, 0.32], row = i+1, col = j+1)
        fig.update_layout(height = 700)
        fig.update_layout(titlefont = dict(size = 20), sliders = sliders)

        figw = go.FigureWidget(fig)
        display(figw)

class Plottingmodule():
    def __init__(self, param = params, q0s = np.linspace(0,0.1,11).tolist(), qus = np.linspace(0, 0.2, 11).tolist(), ρs = [0.5, 1]):
        self.params = {}
        self.params['αŷ'] = params['αk']
        self.params['αẑ'] = params['αz']
        self.params['β̂'] = params['βk']
        self.params['κ̂'] = params['βz']
        self.params['σy'] = params['σy']
        self.params['σz'] = params['σz']
        self.params['δ'] = params['δ']

        self.params['ρ1'] = params['ρ1']
        # self.ρ2 = params['ρ2']
        self.params['z̄'] = params['z̄']
        self.params['σ'] = params['σ']
        self.params['a'] = params['a']
        self.params['b'] = params['b']
        self.params['d'] = params['d']

        self.params['zrange'] = [-2.5, 2.5]
        self.Dz = 0.01
        self.params['zr'] = params['zr']
        self.params['zl'] = params['zl']
        
        if not isinstance(q0s, list):
            if isinstance(q0s, (int, float, np.float)):
                q0s = [q0s]
            else:
                q0s = q0s.tolist()
        if np.inf in qus:
            pass
        else:
            qus.append(np.inf)

        if not isinstance(qus, list):
            if isinstance(qus, (int, float, np.float)):
                qus = [qus]
            else:
                qus = qus.tolist()

        if not isinstance(ρs, list):
            if isinstance(ρs, (int, float, np.float)):
                ρs = [ρs]
            else:
                ρs = ρs.tolist()
        
        self.q0s_list = sorted(q0s)
        self.qus_list = sorted(qus)
        self.ρ_list = sorted(ρs)
        self.models = {}
        self.models = pickle.load(open('Plottingdata.pickle', "rb", -1))

        x_neg = np.append(np.arange(-2.5, 0, self.Dz), 0)  
        x_pos = np.append(np.arange(0, 2.5, self.Dz), 2.5)
        
        self.x = np.hstack([x_neg, x_pos[1:]])

    def dumpdata(self):
        data = {}
        print(self.models)
        for q0s in self.q0s_list:
            for qus in self.qus_list:
                ρ_restricted = q0s ** 2 / norm(self.params['σz']) ** 2
                for ρ in self.ρ_list:
                    data[q0s, qus, ρ] = {}
                    data[q0s, qus, ρ]['ρ'] = ρ_restricted * ρ
                    
                    data[q0s, qus, ρ]['driftz'] = self.models[q0s, qus, ρ].driftz
                    for s in ['shock1', 'shock2', 'ambiguity1', 'ambiguity2', 'misspec1', 'misspec2']:
                        print(getattr(self.models[q0s, qus, ρ], s))
                        temp = getattr(self.models[q0s, qus, ρ], s)
                        data[q0s, qus, ρ][s] = {}
                        data[q0s, qus, ρ][s]['q10'] = temp['q10'][:400]
                        data[q0s, qus, ρ][s]['q50'] = temp['q50'][:400]
                        data[q0s, qus, ρ][s]['q90'] = temp['q90'][:400]
                    # data[q0s, qus, ρ]['shock1'] = self.models[q0s, qus, ρ].shock1
                    # data[q0s, qus, ρ]['shock2'] = self.models[q0s, qus, ρ].shock2
                    # data[q0s, qus, ρ]['ambiguity1'] = self.models[q0s, qus, ρ].ambiguity1
                    # data[q0s, qus, ρ]['ambiguity2'] = self.models[q0s, qus, ρ].ambiguity2
                    # data[q0s, qus, ρ]['misspec1'] = self.models[q0s, qus, ρ].misspec1
                    # data[q0s, qus, ρ]['misspec2'] = self.models[q0s, qus, ρ].misspec2

        with open('Plottingdata.pickle', "wb") as file_:
                    pickle.dump(data, file_, -1)
 
    def driftIntPlot(self, q0s = None, qus = None):
        fig = go.Figure()
        base = None
        if isinstance(q0s,  (int, float, np.float)): # plot along q0s
            q_list = self.qᵤₛ_list
            for qus in q_list:
                model = self.models[q0s, qus, 1]
                if base is None:
                    base = True
                    fig.add_trace(go.Scatter(x = self.x  - params['z̄'], y = self.params['αẑ'] - self.params['κ̂'] * self.x  - self.params['z̄'],
                        name = 'Baseline Model', 
                        line = dict(color = 'black', dash = 'solid', width = 3), showlegend = True))
                if qus == q_list[int(0.3 * len(q_list))]:
                    fig.add_trace(go.Scatter(x = self.x - params['z̄'], y = model['driftz'], 
                        name = r'$q_{{u,s}} = {}$'.format(qus),
                        line = dict(color = '#1f77b4', dash = 'dash', width = 3), legendgroup = 'Current Line', showlegend = True, visible = True))
                elif qus == np.inf:
                    fig.add_trace(go.Scatter(x = self.x - params['z̄'], y = model['driftz'], 
                        name = 'Worst Case',
                        line = dict(color = '#1f77b4', dash = 'dash', width = 3), legendgroup = 'Current Line', showlegend = True, visible = False)) 
                else:
                    fig.add_trace(go.Scatter(x = self.x - params['z̄'], y = model['driftz'], 
                        name = r'$q_{{u,s}} = {}$'.format(qus),
                        line = dict(color = '#1f77b4', dash = 'dash', width = 3), legendgroup = 'Current Line', showlegend = True, visible = False)) 

            fig.update_layout(title = r"$\text{{Growth rate drift comparisions with }}q_{{0,s}} = {:.2f}$".format(q0s), titlefont = dict(size = 20), height = 700)
            
            steps = []
            for i in range(1, len(q_list) + 1):
                if i == len(q_list):
                    label = 'Worst Case'
                else:
                    label =  '{:.2f}'.format(q_list[i-1])
                step = dict(
                    method = 'restyle',
                    args = ['visible', [False] * len(fig.data)],
                    label = label
                )
                step['args'][1][0] = True
                step['args'][1][i] = True
                
                steps.append(step)
            sliders = [dict(active = int(0.3 * len(q_list)),
                        currentvalue = {"prefix": "qus: "},
                        pad = {"t": len(q_list) },
                        steps = steps, y = -0.1)]

        elif isinstance(qus, (int, float, np.float)):
            q_list = self.q0s_list
            for q0s in q_list:
                model = self.models[q0s, qus, 1]
                if base is None:
                    base = True
                    fig.add_trace(go.Scatter(x = self.x  - params['z̄'], y = self.params['αẑ'] - self.params['κ̂'] * self.x  - self.params['z̄'],
                        name = 'Baseline Model', 
                        line = dict(color = 'black', dash = 'solid', width = 3), showlegend = True))
                if q0s == q_list[int(0.3 * len(q_list))]:
                    fig.add_trace(go.Scatter(x = self.x - params['z̄'], y = model['driftz'], 
                        name = r'$q_{{0,s}} = {}$'.format(q0s),
                        line = dict(color = '#1f77b4', dash = 'dash', width = 3), legendgroup = 'Current Line', showlegend = True, visible = True))
                else:
                    fig.add_trace(go.Scatter(x = self.x - params['z̄'], y = model['driftz'], 
                        name = r'$q_{{0,s}} = {}$'.format(q0s),
                        line = dict(color = '#1f77b4', dash = 'dash', width = 3), legendgroup = 'Current Line', showlegend = True, visible = False)) 
            fig.update_layout(title = r"$\text{{Growth rate drift comparisions with }}q_{{u,s}} = {:.2f}$".format(qus), titlefont = dict(size = 20), height = 600)

            steps = []
            for i in range(1, len(q_list) + 1):
                label =  '{:.2f}'.format(q_list[i-1])
                step = dict(
                    method = 'restyle',
                    args = ['visible', [False] * len(fig.data)],
                    label = label
                )
                step['args'][1][0] = True
                step['args'][1][i] = True
                
                steps.append(step)
            
            sliders = [dict(active = int(0.3 * len(q_list)),
                        currentvalue = {"prefix": "q0s: "},
                        pad = {"t": len(q_list) },
                        steps = steps, y = -0.1)]

        fig.update_layout(xaxis = go.layout.XAxis(title=go.layout.xaxis.Title(
                                            text="z", font=dict(size=16)),
                                                tickfont=dict(size=12), showgrid = False),
                        yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(
                                            text="μz", font=dict(size=16)),
                                                tickfont=dict(size=12), showgrid = False),
                        sliders = sliders
                            )

        fig['layout']['yaxis{}'.format(1)].update(showgrid = False)
        fig['layout']['xaxis{}'.format(1)].update(showgrid = False)
        # fig.update_layout(legend = dict(orientation = 'h', y = 1.1))
        fig.update_xaxes(range = [-0.5, 0.5])
        fig.update_yaxes(range = [-0.025, 0.01])
        figw = go.FigureWidget(fig)
        display(figw)

    def shocksIntPlot(self, q0s = None, qus = None):
        x = np.arange(0, 1000.1 ,0.1)
        x = x[:400]
        fig = make_subplots(rows = 2, cols = 3, print_grid = False, vertical_spacing = 0.08,
                    subplot_titles = (('first shock', 'ambiguity price, first shock', 'misspecification price, first shock',
                                    'second shock', 'ambiguity price, second shock', 'misspecification price, second shock')))
        if isinstance(q0s,  (int, float, np.float)): 
            q_list = self.qᵤₛ_list[:-1]
            for qus in q_list:
                model = self.models[q0s, qus, 1]
                if qus == q_list[int(0.3 * len(q_list))]:
                    vis = True
                else:
                    vis = False
                for i,s in enumerate(['shock1', 'ambiguity1', 'misspec1', 'shock2', 'ambiguity2', 'misspec2']):
                    # print(i % 3 + 1, int((i+3)/ 3))
                    
                    fig.add_trace(go.Scatter(x = x, y = model[s]['q10'], 
                        line = dict(color = 'red', dash = 'dot', width = 3), showlegend = False, legendgroup='.1 decile', name = '.1 decile',
                        visible = vis), col = i % 3 + 1, row = int((i+3) / 3))
                                
                    fig.add_trace(go.Scatter(x = x, y = model[s]['q50'], 
                        line = dict(color = 'Black', dash = 'solid', width = 3), showlegend = False, legendgroup='median', name='median',
                        visible = vis), col = i % 3 + 1, row = int((i+3) / 3))
                    fig.add_trace(go.Scatter(x = x, y = model[s]['q90'], 
                        line = dict(color = '#1f77b4', dash = 'dash', width = 3), showlegend = False, legendgroup='.9 decile', name='.9 decile',
                        visible = vis), col = i % 3 + 1, row = int((i+3) / 3))

            fig.update_layout(title = r"$\text{{Shock Price Elasticities Decomposition with }}q_{{0,s}} = {:.2f}$".format(q0s), titlefont = dict(size = 20), height = 700)
            steps = []
            for i in range(len(q_list)):
                if i == len(q_list):
                    label = 'Worst Case'
                else:
                    label =  '{:.2f}'.format(q_list[i])
                step = dict(
                    method = 'restyle',
                    args = ['visible', [False] * len(fig.data)],
                    label = label
                )
                for j in range(18):
                    step['args'][1][i * 18 + j] = True
                
                steps.append(step)
            sliders = [dict(active = int(0.3 * len(q_list)),
                        currentvalue = {"prefix": "qus: "},
                        pad = {"t": len(q_list) },
                        steps = steps, y = -0.15)]

        elif isinstance(qus,  (int, float, np.float)): 
            q_list = self.q0s_list
            for q0s in q_list:
                model = self.models[q0s, qus, 1]
                if q0s == q_list[int(0.3 * len(q_list))]:
                    vis = True
                else:
                    vis = False
                for i,s in enumerate(['shock1', 'ambiguity1', 'misspec1', 'shock2', 'ambiguity2', 'misspec2']):
                    # print(i % 3 + 1, int((i+3)/ 3))
                    
                    fig.add_trace(go.Scatter(x = x, y = model[s]['q10'], 
                        line = dict(color = 'red', dash = 'dot', width = 3), showlegend = False, legendgroup='.1 decile', name = '.1 decile',
                        visible = vis), col = i % 3 + 1, row = int((i+3) / 3))
                                
                    fig.add_trace(go.Scatter(x = x, y = model[s]['q50'], 
                        line = dict(color = 'Black', dash = 'solid', width = 3), showlegend = False, legendgroup='median', name='median',
                        visible = vis), col = i % 3 + 1, row = int((i+3) / 3))
                    fig.add_trace(go.Scatter(x = x, y = model[s]['q90'], 
                        line = dict(color = '#1f77b4', dash = 'dash', width = 3), showlegend = False, legendgroup='.9 decile', name='.9 decile',
                        visible = vis), col = i % 3 + 1, row = int((i+3) / 3))

            fig.update_layout(title = r"$\text{{Shock Price Elasticities Decomposition with }}q_{{u,s}} = {:.2f}$".format(qus), titlefont = dict(size = 20), height = 700)
            steps = []
            for i in range(len(q_list)):
                label =  '{:.2f}'.format(q_list[i])
                step = dict(
                    method = 'restyle',
                    args = ['visible', [False] * len(fig.data)],
                    label = label
                )
                for j in range(18):
                    step['args'][1][i * 18 + j] = True
                
                steps.append(step)
            sliders = [dict(active = int(0.3 * len(q_list)),
                        currentvalue = {"prefix": "q0s: "},
                        pad = {"t": len(q_list) },
                        steps = steps, y = -0.15)]

        for i in range(6):
                
            fig['layout']['yaxis{}'.format(i+1)].update(showgrid = False)
            fig['layout']['xaxis{}'.format(i+1)].update(showgrid = False)
        
        for i in range(3,6):
            fig['layout']['xaxis{}'.format(i+1)].update(title=go.layout.xaxis.Title(
                                        text="Horizon(quarters)", font=dict(size=16)), showgrid = False)
                
            
        for i in range(3):
            for j in range(3):
                fig.update_xaxes(range = [0, 40], row = i+1, col = j+1)
                fig.update_yaxes(range = [0, 0.32], row = i+1, col = j+1)
        fig.update_layout(height = 700)
        fig.update_layout(titlefont = dict(size = 20), sliders = sliders)

        figw = go.FigureWidget(fig)
        display(figw)

    def Figure2(self, q_list = np.linspace(0,0.15)):
        
        [κgrid, βgrid] = np.meshgrid(np.arange(0,0.5,0.001), np.arange(-3,3, 0.005))
        σinv = inv(self.params['σ'])
        η1 = σinv[0,0] * (βgrid - self.params['β̂']) + σinv[0,1] * (self.params['κ̂'] - κgrid)
        η2 = σinv[1,0] * (βgrid - self.params['β̂']) + σinv[1,1] * (self.params['κ̂'] - κgrid)
        data = []
        q_list = sorted(q_list)
        for q in q_list:
            if q == 0:
                data.append([])
            else:
                lhs = 0.5 * (η1 ** 2 + η2 ** 2) + (q ** 2 / norm(self.params['σz'] ** 2)) * (-self.params['κ̂'] + self.params['σz'][0] * η1 + self.params['σz'][1] * η2)
                cs = plt.contour(κgrid, βgrid, lhs, levels = 0)
                dta = cs.allsegs[1][0]
                data.append([dta[:,0], dta[:,1]])
        plt.close()
        
        fig = go.Figure()
        base = None
        l = len(q_list)
        for i in range(len(data)):
            if base is None:
                
                if len(data[i]) == 0:
                    
                    base_x = np.mean(data[i+1][0][1:])
                    base_y = np.mean(data[i+1][1][1:])
                    fig.add_trace(go.Scatter(x = [base_x], y = [base_y], visible = True, name = 'Baseline model',
                                            showlegend = True, legendgroup = 'Baseline model'))
                else:
                    base_x = np.mean(data[i][0][1:])
                    base_y = np.mean(data[i][1][1:])
                    fig.add_trace(go.Scatter(x = [base_x], y = [base_y], visible = True, name = 'Baseline model',
                                            showlegend = True, legendgroup = 'Baseline model'))
                base = 1
            else:
                if i == int(l*0.3):
                    fig.add_trace(go.Scatter(x = np.array(data[i][0][:]), y = np.array(data[i][1][:]), visible = True, name = 'q = {:.3f}'.format(q_list[i]),
                                            showlegend = True, legendgroup = 'RE'))
                else:
                    fig.add_trace(go.Scatter(x = np.array(data[i][0][:]), y = np.array(data[i][1][:]), visible = False, name = 'q = {:.3f}'.format(q_list[i]),
                                            showlegend = True, legendgroup = 'RE'))
        steps = []
        for i in range(len(q_list)):
            if i == 0:
                label = 'baseline'
            else:
                label =  'q = {:.3f}'.format(q_list[i])
            step = dict(
                method = 'restyle',
                args = ['visible', [False] * len(fig.data)],
                label = label
            )
            step['args'][1][0] = True
            step['args'][1][i] = True
            
            steps.append(step)
        
        sliders = [dict(active = int(0.3 * l),
                    currentvalue = {"prefix": "q： "},
                    pad = {"t": len(q_list)},
                    steps = steps)]
        fig.update_layout(title = "Parameter Contours for (βz, βk) holding Relative Entropy Fixed", titlefont = dict(size = 20), height = 800,
                            xaxis = go.layout.XAxis(title=go.layout.xaxis.Title(
                                                text="βz", font=dict(size=16)),
                                                    tickfont=dict(size=12), showgrid = False),
                            yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(
                                                text="βk", font=dict(size=16)),
                                                    tickfont=dict(size=12), showgrid = False),
                            sliders = sliders
                            )
        fig.update_xaxes(range = [min(data[-1][0]), max(data[-1][0])])
        fig.update_yaxes(range = [min(data[-1][1]), max(data[-1][1])])
        
        fig.show()

    def DriftComparison(self, q0s = [0.05, 0.1], ρs = 1, qus = [0.1, 0.2, np.Inf]): # 
        if type(q0s) == list:  # plot against q
            titles = [r"$\sf q_{{s,0}}: {}$".format(q) for q in q0s]
            fig = make_subplots(rows = 1, cols = len(q0s), print_grid = False, subplot_titles = titles)
            x = self.x

            for i, q0 in enumerate(q0s):
                if i == 0:
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.2, ρs]['driftz'], 
                                                name = 'qᵤₛ = .2', legendgroup = 'qᵤₛ = .2', line = dict(color = 'green', dash = 'dashdot', width = 3), showlegend = True),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.1, ρs]['driftz'], 
                                            name = 'qᵤₛ = .1',legendgroup = 'qᵤₛ = .1', line = dict(color = '#1f77b4', dash = 'solid', width = 3), showlegend = True),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, np.inf, ρs]['driftz'], 
                                                name = 'Worst Case Scenario', legendgroup = 'Worst Case Scenario', line = dict(color = 'red', dash = 'dot', width = 3), showlegend = True),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.params['αẑ'] - self.params['κ̂']* (x - self.params['z̄']), 
                                                name = 'Baseline Model', legendgroup = 'Baseline Model', line = dict(color = 'black', dash = 'solid', width = 3), showlegend = True),
                            row = 1, col = i +1)
                else:
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.2, ρs]['driftz'], 
                                                name = 'qᵤₛ = .2', legendgroup = 'qᵤₛ = .2', line = dict(color = 'green', dash = 'dashdot', width = 3), showlegend = False),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.1, ρs]['driftz'], 
                                            name = 'qᵤₛ = .1',legendgroup = 'qᵤₛ = .1', line = dict(color = '#1f77b4', dash = 'solid', width = 3), showlegend = False),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, np.inf, ρs]['driftz'], 
                                                name = 'Worst Case Scenario', legendgroup = 'Worst Case Scenario', line = dict(color = 'red', dash = 'dot', width = 3), showlegend = False),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.params['αẑ'] - self.params['κ̂']* (x - self.params['z̄']), 
                                                name = 'Baseline Model', legendgroup = 'Baseline Model', line = dict(color = 'black', dash = 'solid', width = 3), showlegend = False),
                            row = 1, col = i +1)
            
            fig.update_layout(title = r"$\text{Growth rate drift comparisions with different } \sf q_{{0,s}}$", titlefont = dict(size = 20))

        else: # plot against rho

            rho = self.models[q0s, 0.1, 1]['ρ']
            titles = [r"$\text{{Relaxed }}\rho_2 = {{\frac{{\sf q_{{s,0}}^2}}{{2|\sigma^2|}}}}$ = {:.2f}".format(rho * ρs[0]),  
                        r"$\text{{Restricted }}\rho_2 = {{\frac{{\sf q_{{s,0}}^2}}{{|\sigma^2|}}}}$ = {:.2f}".format(rho * ρs[1])]
            fig = make_subplots(rows = 1, cols = len(ρs), print_grid = False, subplot_titles = titles)
            q0 = q0s
            x = self.x
            for i, rs in enumerate(ρs):
                
                if i == 0:
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.2, rs]['driftz'], 
                                                name = 'qᵤₛ = .2', legendgroup = 'qᵤₛ = .2', line = dict(color = 'green', dash = 'dashdot', width = 3), showlegend = True),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.1, rs]['driftz'], 
                                            name = 'qᵤₛ = .1',legendgroup = 'qᵤₛ = .1', line = dict(color = '#1f77b4', dash = 'solid', width = 3), showlegend = True),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, np.inf, rs]['driftz'], 
                                                name = 'Worst Case Scenario', legendgroup = 'Worst Case Scenario', line = dict(color = 'red', dash = 'dot', width = 3), showlegend = True),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.params['αẑ'] - self.params['κ̂']* (x - self.params['z̄']), 
                                                name = 'Baseline Model', legendgroup = 'Baseline Model', line = dict(color = 'black', dash = 'solid', width = 3), showlegend = True),
                            row = 1, col = i +1)
                else:
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.2, rs]['driftz'], 
                                                name = 'qᵤₛ = .2', legendgroup = 'qᵤₛ = .2', line = dict(color = 'green', dash = 'dashdot', width = 3), showlegend = False),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, 0.1, rs]['driftz'], 
                                            name = 'qᵤₛ = .1',legendgroup = 'qᵤₛ = .1', line = dict(color = '#1f77b4', dash = 'solid', width = 3), showlegend = False),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.models[q0, np.inf, rs]['driftz'], 
                                                name = 'Worst Case Scenario', legendgroup = 'Worst Case Scenario', line = dict(color = 'red', dash = 'dot', width = 3), showlegend = False),
                            row = 1, col = i +1) 
                    fig.add_trace(
                        go.Scatter(x = x - self.params['z̄'], y = self.params['αẑ'] - self.params['κ̂']* (x - self.params['z̄']), 
                                                name = 'Baseline Model', legendgroup = 'Baseline Model', line = dict(color = 'black', dash = 'solid', width = 3), showlegend = False),
                            row = 1, col = i +1)
            fig.update_layout(title = r"$\text{Growth rate drift comparisions betweeen restricted and unrestricted } \rho_2$", titlefont = dict(size = 20))

        for i in range(2):
                
            fig['layout']['yaxis{}'.format(i+1)].update(showgrid = False)
            fig['layout']['xaxis{}'.format(i+1)].update(showgrid = False)
            fig['layout']['xaxis{}'.format(i+1)].update(title=go.layout.xaxis.Title(
                                        text="z", font=dict(size=16)), showgrid = False)
                
        fig['layout']['yaxis1'].update(title=go.layout.yaxis.Title(
                                        text="μz", font=dict(size=16)), showgrid = False)
            
            
        fig.update_xaxes(range = [-0.5, 0.5], row = 1, col = 1)
        fig.update_yaxes(range = [-0.025, 0.01], row = 1, col = 1)
        fig.update_xaxes(range = [-0.5, 0.5], row = 1, col = 2)
        fig.update_yaxes(range = [-0.025, 0.01], row = 1, col = 2)
        fig.show()
    
    def Figure6(self, q0s = [0.05, 0.1], qus = 0.2):
        x = np.arange(0, 1000.1 ,0.1)
        x = x[:400]
        fig = make_subplots(rows = 2, cols = 2, print_grid = False, vertical_spacing = 0.08,
                    subplot_titles = (('first shock with qₛₒ = {:.2f}'.format(q0s[0]), 'first shock with qₛₒ = {:.2f}'.format(q0s[1]),\
                                    'second shock with qₛₒ = {:.2f}'.format(q0s[0]), 'second shock with qₛₒ = {:.2f}'.format(q0s[1]))))
        for i, s in enumerate(['shock1', 'shock2']):
            for j, q in enumerate(q0s):
                model = self.models[q, qus, 1]
                fig.add_trace(go.Scatter(x = x, y = model[s]['q10'], 
                                line = dict(color = 'red', dash = 'dot', width = 3), showlegend = False, legendgroup='.1 decile', name = '.1 decile'),
                                row = i + 1, col = j + 1) 
                fig.add_trace(go.Scatter(x = x, y = model[s]['q50'], 
                                line = dict(color = 'Black', dash = 'solid', width = 3), showlegend = False, legendgroup='median', name='median'),
                                row = i + 1, col = j + 1) 
                fig.add_trace(go.Scatter(x = x, y = model[s]['q90'], 
                                line = dict(color = '#1f77b4', dash = 'dash', width = 3), showlegend = False, legendgroup='.9 decile', name='.9 decile'),
                                row = i + 1, col = j + 1) 
        fig.data[0]['showlegend'] = True
        fig.data[1]['showlegend'] = True
        fig.data[2]['showlegend'] = True
        for i in range(4):
                
            fig['layout']['yaxis{}'.format(i+1)].update(showgrid = False)
            fig['layout']['xaxis{}'.format(i+1)].update(showgrid = False)
        for i in range(2,4):
            fig['layout']['xaxis{}'.format(i+1)].update(title=go.layout.xaxis.Title(
                                        text="Horizon(quarters)", font=dict(size=16)), showgrid = False)

        for i in range(2):
            for j in range(2):
                fig.update_xaxes(range = [0, 40], row = i+1, col = j+1)
                fig.update_yaxes(range = [0, 0.32], row = i+1, col = j+1)
        fig.update_layout(height = 700)
        fig.update_layout(title = r"$\text{Shock price elasticities with different }\sf q_{s,0}$", titlefont = dict(size = 20))
        fig.show()

    def Figure7(self, q0s = 0.1, qus = 0.2):
        x = np.arange(0, 1000.1 ,0.1)
        x = x[:400]
        fig = make_subplots(rows = 2, cols = 2, print_grid = False, vertical_spacing = 0.08,
                    subplot_titles = (('ambiguity price for the first shock', 'misspecification price for the first shock',\
                                    'ambiguity price for the second shock', 'misspecification price for the second shock')))
        for i, s in enumerate(['ambiguity1', 'ambiguity2', 'misspec1', 'misspec2']):
            model = self.models[q0s, qus, 1]
            fig.add_trace(go.Scatter(x = x, y = model[s]['q10'], 
                            line = dict(color = 'red', dash = 'dot', width = 3), showlegend = False, legendgroup='.1 decile', name = '.1 decile'),
                        row = (i+1) % 2 + 1, col = int((i+2) / 2))
            fig.add_trace(go.Scatter(x = x, y = model[s]['q50'], 
                            line = dict(color = 'Black', dash = 'solid', width = 3), showlegend = False, legendgroup='median', name='median'),
                        row = (i+1) % 2 + 1, col = int((i+2) / 2))
            fig.add_trace(go.Scatter(x = x, y = model[s]['q90'], 
                            line = dict(color = '#1f77b4', dash = 'dash', width = 3), showlegend = False, legendgroup='.9 decile', name='.9 decile'),
                        row = (i+1) % 2 + 1, col = int((i+2) / 2))

        fig.data[0]['showlegend'] = True
        fig.data[1]['showlegend'] = True
        fig.data[2]['showlegend'] = True
        for i in range(4):
                
            fig['layout']['yaxis{}'.format(i+1)].update(showgrid = False)
            fig['layout']['xaxis{}'.format(i+1)].update(showgrid = False)

        for i in range(2,4):
            fig['layout']['xaxis{}'.format(i+1)].update(title=go.layout.xaxis.Title(
                                        text="Horizon(quarters)", font=dict(size=16)), showgrid = False)
                
            
        for i in range(2):
            for j in range(2):
                fig.update_xaxes(range = [0, 40], row = i+1, col = j+1)
                fig.update_yaxes(range = [0, 0.32], row = i+1, col = j+1)
        fig.update_layout(height = 700)
        fig.update_layout(title = "Shock price elasticities Decomposition", titlefont = dict(size = 20))
        fig.show()


if __name__ == "__main__":
    print('-----------------------------------Starting-------------------------------------------')
    start_time = datetime.datetime.now()
    # s = SturcturedModel(params, 0.2, 0)
    # s.ApproxBound()
    # print("q0s = {}; qus = {}; rho2 = {}; bounds are {}, {}".format(s.q0s, s.qᵤₛ, s.ρ2, s.dvl, s.dvr))
    # # err = s.CalibratingTheta(0.2, 1.0)
    # # print('The error is {}'.format(err))
    # s.solvetheta(None)
    # # s.CalibratingTheta(0.9864243155699565, 1.0)
    # print('θ = {}'.format(s.θ))
    # print('Solved: {}'.format(s.status))
    # s.HL(calHL = True)
    # s.Drift()
    # s.ExpectH()
    # print(np.linspace(0,0.1,11).tolist())
    # print(np.linspace(0,0.2,11).tolist())
    # s = TenuousModel(params, np.linspace(0,0.1,11).tolist(), np.linspace(0, 0.2, 11).tolist(), [0.5, 1])
    # s.Figure2()
    p = Plottingmodule()
    # p.driftIntPlot(q0s = 0.1)
    # p.shocksIntPlot(q0s = 0.1)
    # p.Figure2()
    # p.DriftComparison()
    p.Figure6()
    p.Figure7()
    # p.dumpdata()
    # s.driftIntPlot(0.1,0.2,1)
    # print(s.models.keys())
    # s.Figure2()
    # s.DriftComparison(0.1, [0.5,1])
    # s.Figure6()
    # s.Figure7()
    print('Time spent is {}'.format(datetime.datetime.now()- start_time))