# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:52:32 2022

@author: Mohamed KH
"""


import numpy
import scipy.optimize
from numpy.linalg import norm

def lagrange_Eq (alpha,*args):
    H= args[0]
    mid= numpy.dot(alpha.T,H)
    onesMat= numpy.ones((alpha.shape[0],1)) 
    LD= 1/2*numpy.dot(mid,alpha) - numpy.dot(alpha.T,onesMat)
    gradLD= numpy.dot(H,alpha)-1
    return LD[0],gradLD

def primal_Loss (Dt,Wopt,C,LTR):
    tmr=0
    z = numpy.array([1 if LTR[_] else -1 for _ in range(Dt.shape[1])])
    tmp = 1-z*(numpy.dot(Wopt.T, Dt))
    tmp1 = tmp>0
    tmp2 = tmp[tmp1]
    tmr = numpy.sum(tmp2)
    
    j= 1/2* norm(Wopt)**2 + C*tmr
    print("primal loss:")
    print(j)
    return j


class L_SVM:
    def __init__(self,K,C,prior):
        self.K = K
        self.C = C
        self.Wopt =0
        self.Dt=0
        self.ltr =0
        self.f=0
        self.prior = prior
    
    def train(self, DTR, LTR):
        kas= numpy.full(shape=(1,DTR.shape[1]),fill_value=self.K)
        self.Dt = numpy.vstack([DTR,kas])
        G = numpy.dot(self.Dt.T,self.Dt)
        H = numpy.zeros(G.shape)
        self.ltr=LTR

        zj = numpy.array([1 if LTR[_] else -1 for _ in range(self.Dt.shape[1])])
        zi = numpy.ones([zj.shape[0],zj.shape[0]])
        zi[:] = zj
        zi = zi.T
        H = zi*zj*G
        alpha = numpy.zeros((DTR.shape[1],1))
        
        C1 = (self.C * self.prior) / (DTR[:, LTR == 1].shape[1] / DTR.shape[1])
        C0 = (self.C * (1 - self.prior)) / (DTR[:, LTR == 0].shape[1] / DTR.shape[1])
        alphaBounds = [(0, C0) if LTR[_] == 0 else (0, C1) for _ in range(LTR.shape[0])]
        
        x,self.f,d=scipy.optimize.fmin_l_bfgs_b(lagrange_Eq,alpha,args=(H,1),approx_grad=False,bounds=alphaBounds,factr=1.0) 
        self.Wopt = numpy.sum(x*zj*self.Dt,axis=1)
       

    def evaluate(self, DTE):
        predictedCls=numpy.zeros(DTE.shape[1],dtype=int)
        kos= numpy.full(shape=(1,DTE.shape[1]),fill_value=self.K)
        Dtt = numpy.vstack([DTE,kos])
        S = numpy.dot(self.Wopt.T,Dtt)
        predictedCls = S>0
        
        
        return predictedCls,S
    



