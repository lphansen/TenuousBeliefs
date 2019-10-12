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

params = {}
params['q'] = 0.05

params['αŷ'] = 0.386
params['αẑ'] = 0
params['β̂'] = 1
params['κ̂'] = 0.019
params['σy'] = np.array([[0.488], [0]])
params['σz'] = np.array([[0.013], [0.028]])
params['δ'] = 0.002

params['ρ1'] = 0

params['ρ2'] = params['q'] ** 2 / norm(params['σz']) ** 2
params['z̄'] = params['αẑ'] / params['κ̂']
params['σ'] = np.vstack([params['σy'].T, params['σz'].T ]) 
params['a'] = norm(params['σz']) ** 2 /  det(params['σ'] ) ** 2
params['b'] = - np.squeeze(params['σy'].T.dot(params['σz'])) /  det(params['σ'] ) ** 2
params['d'] = norm(params['σy']) ** 2 /  det(params['σ'] ) ** 2

ρ2_default = params['ρ2']

class SturcturedModel():

    def __init__(self, params, q₀ₔ, qᵤₔ, ρ2, dv0guess):
        self.q_target = q_target
        self.θ = None
        self.αŷ = params['αŷ']
        self.αẑ = params['αẑ']
        self.β̂ = params['β̂']
        self.κ̂ = params['κ̂']
        self.σy = params['σy']
        self.σz = params['σz']
        self.δ = params['δ']

        self.ρ1 = params['ρ1']
        self.ρ2 = ρ2
        self.z̄ = params['z̄']
        self.σ = params['σ']
        self.a = params['a']
        self.b = params['b']
        self.d = params['d']
        self.q₀ₔ = q₀ₔ
        self.qᵤₔ = qᵤₔ

        self.zrange = [-2.5, 2.5]
        self.Dz = 0.01
        
        self.x = None # this is the z grid
        self.y = None

        self.s1 = None
        self.s2 = None

    def HJBODE(z, v, θ):
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
            min_val = mined(z, v1)
            temp = np.zeros(len(v1))
            for i in range(len(v1)):
        #         print(np.array([[0.01],[v1[i]]]))
        #         print(np.array([0.01, v1[i]]))
                temp[i] = np.array([0.01, v1[i]]).dot(self.σ).dot(self.σ.T).dot(np.array([[0.01],[v1[i]]]))

            return np.vstack((v1,
                            2 / norm(self.σz) ** 2 * (self.δ * v0 - min_val + 1 / (2 * θ) *  temp )))

    def mined(z, dv):
        
        A = 0.5 * self.a
        C0 = (self.ρ1 + self.ρ2 * (z - self.z̄)) * (self.αẑ - self.κ̂ * (z - self.z̄)) + norm(self.σz) ** 2 / 2 * self.ρ2 - self.q₀ₔ ** 2 / 2
        C1 = (self.ρ1 + self.ρ2 * (z - self.z̄))
        C2 = 0.5 * self.d
        D = self.b ** 2 / (2 * A) - 2 * C2
        E = (100 * dv - self.b / (2*A)) ** 2
        
        AA = E * b **2 - 4 * A * E * C2 - D ** 2
        BB = 2 * C1 * D - 4 * A * E * C1
        CC = - 4 * A * C0 * E - C1 ** 2
        
        s21 = ( -BB + np.sqrt(BB ** 2 - 4 * AA * CC)) / (2 * AA)
        s22 = ( -BB - np.sqrt(BB ** 2 - 4 * AA * CC)) / (2 * AA)
        
        mined1 = np.squeeze(0.01 * (self.αŷ + self.β̂ * (z-self.z̄) + S1(s21, z)) + dv * (self.αẑ - self.κ̂ * (z - self.z̄) + s21))
        mined2 = np.squeeze(0.01 * (self.αŷ + self.β̂ * (z-self.z̄) + S1(s22, z)) + dv * (self.αẑ - self.κ̂ * (z - self.z̄) + s22))
        
        if isinstance(mined1, (int, float, np.float)):
            res = min(mined1, mined2)
            return (res, s21 * (mined1 <= mined2) + s22 * (mined1 > mined2))
        else:
            res = np.min(np.vstack([mined1,mined2]), axis = 0)
            return res

    def S1(s2, z):
    
        A = 0.5 * self.a
        B = self.b * s2
        C = 0.5 * self.d * s2 ** 2 + (self.ρ1 + self.ρ2 * (z - self.z̄)) * s2 + (self.ρ1 + self.ρ2 * (z - self.z̄)) * (self.αẑ - self.κ̂ * (z - self.z̄)) + norm(self.σz) ** 2 / 2 * self.ρ2 - self.q₀ₔ ** 2 / 2
        
        return (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

    def ApproxBound():
    
        ν, s1, s2 = sympy.symbols('ν s1 s2')
        f1 = (-self.δ - self.κ̂ + s2) * ν + 0.01 * (self.β̂ + s1)
        f2 = ν * (self.a * s1 + self.b * s2) - 0.01 * (self.b * s1 + self.d * s2 + self.ρ2)
        f3 = 0.5 * (self.a * s1 ** 2 + 2 * self.b * s1 * s2 + self.d * s2 ** 2) + self.ρ2 * (-self.κ̂ + s2 )
        initialGuess = (np.array([0.2, 0.8]), np.array([-0.1, 0.1]), np.array([0, 0]))
        bounds =  np.array(sympy.solvers.solve((f1, f2, f3), (ν, s1, s2))).astype(float)
        self.dvl = max(bounds[:,0])
        self.dvr = min(bounds[:,0])
    
    def ODEsolver(zrange, bdl, bdr, θ):
        def tosolve(z, v):
            return HJBODE(z, v, θ)
        
        def bc(ya, yb):
            return np.array([ya[1] - bdl, yb[1] - bdr])
        
        if abs(zrange[0]) >= abs(zrange[1]):
            temp = bdr
        else:
            temp = bdl
            
        x = np.linspace(zrange[0], zrange[1], 10)
    #     print(x)
    #     print(temp)
        y = np.ones((2,x.size)) * np.array([0, temp])[:,np.newaxis]
    #     print(y)
        res = solve_bvp(tosolve, bc, x, y)
        return res
        
    def MatchODE(θ, zl, zr, dvl, dvr, Dz):
        res = {}
        
        def v0Diff(dv0):
    #         print(type(dv0))
    #         if type(dv0) is list or np.ndarray:
    #             dv0 = dv0[0]
            print('trying dv(0) = {}'.format(dv0))
            negsol = ODEsolver([self.zrange[0], 0], self.dvl, dv0, θ)
            print('For this case, v(0-) = {}'.format(negsol['y'][0,-1]))
            possol = ODEsolver([0, self.zrange[1]], dv0, self.dvr, θ)
            print('For this case, v(0+) = {}'.format(possol['y'][0, 0]))

            diff = negsol['y'][0, -1] - possol['y'][0, 0]
            print('The difference is: {}'.format(diff))
            return diff

        dv0 = np.squeeze(fsolve(v0Diff, self.dv0guess))
    #     return np.squeeze(fsolve(v0Diff, dv0guess))

        print('-----------------------')
        print('dv matched at {}'.format(dv0))
        
        negsol = ODEsolver([zl, 0], dvl, dv0, θ)
        v0 = negsol.y[0,-1]
        v1 = negsol.y[1,-1]
    #     print(type(v1))
        (min_val,_) = mined(-1e-6, v1)
        v2 = 2 / norm(self.σz) ** 2 * (self.δ * v0 - min_val + 1 / (2 * θ) *  np.array([0.01, v1]).dot(self.σ).dot(self.σ.T).dot(np.array([[0.01],[v1]])))
        print("For θ = {}, v(0-) = {}; v'(0-) = {}; v''(0-) = {}".format(θ, v0, v1, v2))
    #     return negsol
        possol = ODEsolver([0, zr], dv0, dvr, θ)
        v0 = possol.y[0, 0]
        v1 = possol.y[1, 0]
        (min_val,_) = mined(1e-6, v1)
        v2 = 2 / norm(self.σz) ** 2 * (self.δ * v0 - min_val + 1 / (2 * θ) *  np.array([0.01, v1]).dot(self.σ).dot(self.σ.T).dot(np.array([[0.01],[v1]])))
        print("For θ = {}, v(0+) = {}; v'(0+) = {}; v''(0+) = {}".format(θ, v0, v1, v2))
        
        x_neg = np.append(np.arange(-2.5, 0, self.Dz), 0)
        negSpline = CubicSpline(negsol.x, negsol.y, axis = 1)
        negSplined = negSpline(x_neg)
        
        x_pos = np.append(np.arange(0, 2.5, self.Dz), 2.5)
        posSpline = CubicSpline(possol.x, possol.y, axis = 1)
        posSplined = posSpline(x_pos)
        
        self.x = np.hstack([x_neg, x_pos[1:]])
        self.y = np.hstack([negSplined, posSplined[:,1:]])
        res['possol'] = possol
        res['negsol'] = negsol
        return res

    def Distortion(sol, θ):
        # Calculate R
        Nz = len(sol['x'])
        s2 = np.zeros(Nz)
        s1 = np.zeros(Nz)
        
        for j in range(Nz):
            (_, s2[j]) = mined(sol['x'][j], sol['y'][1,j])
            s1[j] = S1(s2[j], sol['x'][j])
            
        s = np.vstack([s1,s2])
        r = solve(self.σ, s)
        
        h = -1 / θ * σ.T.dot(np.vstack([np.ones([1,Nz]) * 0.01, sol['y'][1,:]])) + r
        
        rh = np.vstack([r, h])
        
        return (rh, s1, s2)

    def RelativeEntropyUS(ηᵤ ,ηₛ):
        Nz = len(self.x)
        for j in range(Nz):
            if j == 0:
                Q[0, 0] = (self.σz.T.dot(ηᵤ[:,j]) + self.αẑ - self.κ̂ * self.x[j]) / (self.Dz) - 0.5 * norm(self.σz) ** 2 / (self.Dz ** 2)
                Q[0, 1] = -(self.σz.T.dot(ηᵤ[:,j]) + self.αẑ - self.κ̂ * self.x[j]) / (self.Dz) + norm(self.σz) ** 2 / (self.Dz ** 2)
                Q[0, 2] = -0.5 * norm(self.σz) ** 2 / self.Dz ** 2
                
            elif j == Nz - 1:
                Q[j, j] = -(self.σz.T.dot(ηᵤ[:,j]) + self.αẑ - self.κ̂ * self.x[j]) / (self.Dz) - 0.5 * norm(self.σz) ** 2 / (self.Dz ** 2)
                Q[j, j - 1] = (self.σz.T.dot(ηᵤ[:,j]) + self.αẑ - self.κ̂ * self.x[j]) / (self.Dz) + norm(self.σz) ** 2 / (self.Dz ** 2)
                Q[j, j - 2] = -0.5 * norm(self.σz) ** 2 / self.Dz ** 2
                    
            else:
                
                Q[j, j - 1] = (self.σz.T.dot(ηᵤ[:,j]) + self.αẑ - self.κ̂ * self.x[j]) / (2 * self.Dz) - 0.5 * norm(self.σz) ** 2 / (self.Dz ** 2)
                Q[j, j] = norm(self.σz) ** 2 / self.Dz ** 2
                Q[j, j + 1] = -(self.σz.T.dot(ηᵤ[:,j]) + self.αẑ - self.κ̂ * self.x[j]) / (2 * self.Dz) - 0.5 * norm(self.σz) ** 2 / (self.Dz ** 2)
                
        tmp = ηᵤ - ηₛ
        rhs = (tmp[0,:] ** 2 + tmp[1,:] ** 2) / 2
        lhs = Q
        lhs[:,self.x == self.z̄] = 1
        sol = solve(lhs, rhs)
        q = np.sqrt(sol[self.x == self.z̄] * 2)
        return q


class TenuousModel():

    def __init__(self, param = params, q₀ₔ = [0.05, 0.1], qᵤₔ = [0.1, 0.2], ρs = [0.5 * ρ2_default, ρ2_default]):
        self.params = {}
        self.params['αŷ'] = params['αŷ']
        self.params['αẑ'] = params['αẑ']
        self.params['β̂ ']= params['β̂']
        self.params['κ̂ ']= params['κ̂']
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
        
        if not isinstance(q₀ₔ, list)：
            q₀ₔ = [q₀ₔ]
        if np.inf in q₀ₔ:
            pass
        else:
            q₀ₔ.append(np.inf)

        if not isinstance(qᵤₔ, list)：
            qᵤₔ = [qᵤₔ]

        if not isinstance(ρs, list):
            qᵤₔ = [qᵤₔ]
        
        self.q₀ₔ_list = sorted(q₀ₔ)
        self.qᵤₔ_list = sorted(qᵤₔ)
        self.ρ_list = sorted(ρs)
        self.models = {}
        
        for (q₀ₛ, qᵤₔ, ρ) in zip(self.q₀ₔ_list, self.qᵤₔ_list, self.ρ_list):
            self.models[q₀ₛ, qᵤₔ, ρ] = SturcturedModel(self.params, q₀ₛ, qᵤₔ, ρ)

