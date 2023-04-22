# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 00:28:49 2022

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


def mcol (lst):
    return lst.reshape((lst.shape[0],1))

def mrow (lst):
    return lst.reshape((1,lst.shape[0]))

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

def polykernel(x1,x2,d,c):
    t= numpy.dot(x1.T,x2)+c
    
    return t**d
    
def RBF (x1,x2,lamda):
    t = norm(x1-x2)**2
    return numpy.exp(-lamda*t)

class K_SVM:
    def __init__(self,kernel,K,C,d,c,lamda,prior):
        self.K = K
        self.C = C
        self.d = d
        self.c = c
        self.lamda = lamda
        self.kernel = kernel
        self.Wopt =0
        self.Dt=0
        self.ltr =0
        self.f=0
        self.dtr =0
        self.ltr =0
        self.x = 0
        self.prior = prior
    
    def train(self, DTR, LTR):
        self.dtr = DTR
        self.ltr = LTR
        
        H = numpy.zeros((DTR.shape[1],DTR.shape[1]))
        
        zj = numpy.array([1 if LTR[_] else -1 for _ in range(DTR.shape[1])])
        zi = numpy.ones([zj.shape[0],zj.shape[0]])
        zi[:] = zj
        zi = zi.T
        if self.kernel =='RBF' :
            Dist = mcol((DTR ** 2).sum(0)) + mrow((DTR ** 2).sum(0)) - 2 * numpy.dot(DTR.T, DTR)
            kernel = numpy.exp(-self.lamda * Dist) + (self.K ** 2)        
            H = zi*zj* kernel       
        else :
            kernel = ((numpy.dot(DTR.T, DTR) + self.c) ** self.d) + (self.K ** 2)
            H = zi*zj * kernel
            
        H = numpy.reshape(H, (DTR.shape[1],DTR.shape[1]))
        H = zi*zj* H
       
        self.ltr=LTR
        alpha = numpy.zeros((DTR.shape[1],1))
        
        C1 = (self.C * self.prior) / (DTR[:, LTR == 1].shape[1] / DTR.shape[1])
        C0 = (self.C * (1 - self.prior)) / (DTR[:, LTR == 0].shape[1] / DTR.shape[1])
        alphaBounds = [((0, C0) if x == 0 else (0, C1)) for x in LTR.tolist()]
        
        self.x,self.f,o=scipy.optimize.fmin_l_bfgs_b(lagrange_Eq,alpha,args=(H,1),approx_grad=False,bounds=alphaBounds,factr=1.0) 
        self.Wopt = numpy.sum(self.x*zj*self.dtr,axis=1)
           

    def evaluate(self, DTE):
        predictedCls=numpy.zeros(DTE.shape[1],dtype=int)

        SS= numpy.zeros((1,DTE.shape[1]))
        zj = numpy.array([1 if self.ltr[_] else -1 for _ in range(self.dtr.shape[1])])
        if self.kernel =='RBF':
            Dist = mcol((self.dtr ** 2).sum(0)) + mrow((DTE ** 2).sum(0)) - 2 * numpy.dot(self.dtr.T, DTE)
            kernel = numpy.exp(-self.lamda * Dist) + (self.K ** 2)
            
        else:
            kernel = ((numpy.dot(self.dtr.T, DTE) + self.c) ** self.d) + (self.K ** 2)
        
        
        SS = numpy.dot(self.x * zj, kernel)
        predictedCls= SS>0
        return predictedCls,SS
    



