# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 10:46:15 2022

@author: Mohamed KH
"""

import numpy
import scipy.linalg
import math
import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../Data')

def MLCovariance (X,mu):
    mu = mcol(mu)
    CCov = numpy.dot((X-mu), (X-mu).T)
    CCov /= X.shape[1]
    return CCov

def mcol(lst):
    return lst.reshape((lst.shape[0],1))

def mrow (lst):
    return lst.reshape((1,lst.shape[0]))

def logpdf_GAU_ND (X,mu,C):
    Y = numpy.empty([X.shape[0], X.shape[1]])
    sign,logC = numpy.linalg.slogdet(C)
    Cinv = numpy.linalg.inv(C)

    for i in range(X.shape[1]):
        x = X[:,i:i+1]
        M = x.shape[0]
        logN = -M/2 * math.log(2*math.pi)
        logN -= 1/2 *logC
        pram = numpy.dot((x-mu).T,Cinv)
        logN -= 1/2 * numpy.dot(pram,(x-mu))
        
        Y[0,i] = logN[0][0]

    return Y[0]


def logpdf_GMM (X,gmm):
    
    S = numpy.zeros((len(gmm),X.shape[1]))
    for i in range(len(gmm)):
        w= gmm[i][0]
        mu= gmm[i][1]
        cov= gmm[i][2]
        S[i,:] = logpdf_GAU_ND(X, mu, cov) + numpy.log(w)
        
           
    logdens = scipy.special.logsumexp(S,axis=0)
    
    return logdens,S

def EM(S,X):

    psi=0.01
    MDist = scipy.special.logsumexp(S,axis=0)
    
    Gama = numpy.exp(S-MDist)
    Zg = numpy.sum(Gama,axis =1)
    
    Fg=[] 
    Sg=[]
    for i in range(Gama.shape[0]):
        Fg.append(numpy.sum(mrow(Gama[i]) * X,axis=1))
        Sg.append(numpy.dot(mrow(Gama[i]) * X,X.T))
    
    Sg= numpy.array(Sg)
    Fg= numpy.array(Fg)
    Fg = Fg.T
    NMu= Fg/Zg
    NSeg=[]
    
    for i in range(Gama.shape[0]):
        NSeg.append(Sg[i]/Zg[i] - numpy.dot(mcol(NMu[:,i]),mcol(NMu[:,i]).T))
        U, s, _ = numpy.linalg.svd(NSeg[i])
        s[s<psi] = psi
        NSeg[i] = numpy.dot(U, mcol(s)*U.T)
        
    NSeg=  numpy.array(NSeg)
    NW = Zg/ numpy.sum(Zg)
    return NMu, NSeg,NW

def EM_Diag(S,X):
    psi=0.01
    MDist = scipy.special.logsumexp(S,axis=0)
    Gama = numpy.exp(S-MDist)
    Zg = numpy.sum(Gama,axis =1)
    Fg=[] 
    Sg=[]
    for i in range(Gama.shape[0]):
        Fg.append(numpy.sum(mrow(Gama[i]) * X,axis=1))
        Sg.append(numpy.dot(mrow(Gama[i]) * X,X.T))
    
    Sg= numpy.array(Sg) 
    Fg= numpy.array(Fg)
    Fg = Fg.T
    NMu= Fg/Zg
    NSeg=[]
    
    for i in range(Gama.shape[0]):
        NSeg.append(Sg[i]/Zg[i] - numpy.dot(mcol(NMu[:,i]),mcol(NMu[:,i]).T))
        U, s, _ = numpy.linalg.svd(NSeg[i])
        s[s<psi] = psi
        NSeg[i] = numpy.dot(U, mcol(s)*U.T)
        NSeg[i] = NSeg[i] * numpy.eye(NSeg[i].shape[0])
        
    NSeg=  numpy.array(NSeg)
    NW = Zg/ numpy.sum(Zg)
    return NMu, NSeg,NW    
    
def EM_Tied(S,X):
    psi=0.01
    MDist = scipy.special.logsumexp(S,axis=0)
    Gama = numpy.exp(S-MDist)
    Zg = numpy.sum(Gama,axis =1)
    
    Fg=[] 
    Sg=[]
    for i in range(Gama.shape[0]):
        Fg.append(numpy.sum(mrow(Gama[i]) * X,axis=1))
        Sg.append(numpy.dot(mrow(Gama[i]) * X,X.T))
    
    Sg= numpy.array(Sg) 
    Fg= numpy.array(Fg)
    Fg = Fg.T
    NMu= Fg/Zg
    NSeg=[]
    
    
    for i in range(Gama.shape[0]):
        NSeg.append(Sg[i]/Zg[i] - numpy.dot(mcol(NMu[:,i]),mcol(NMu[:,i]).T))
        U, s, _ = numpy.linalg.svd(NSeg[i])
        s[s<psi] = psi
        NSeg[i] = numpy.dot(U, mcol(s)*U.T)* Zg[i]
        
        
    NSeg=  numpy.array(NSeg)
    SegTied = NSeg.sum(axis =0)/ Gama.shape[0]
    NewSegTied = []
    for i in range(Gama.shape[0]):
        NewSegTied.append(SegTied)
    NewSegTied= numpy.array(NewSegTied)
    
    NW = Zg/ numpy.sum(Zg)
    return NMu, NewSegTied,NW    
    
    

def splitGmm (X,GMM_1,Type):
    newGMM=[]
    alpha = 0.1
    for i in range(len(GMM_1)):
        wg = GMM_1[i][0]
        mug = GMM_1[i][1]
        sig = GMM_1[i][2]
        U, s, Vh = numpy.linalg.svd(sig)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        newGMM.append((wg/2,mug+d,sig))
        newGMM.append((wg/2,mug-d,sig))
        
    
    calculatedLogDensity, S= logpdf_GMM(X,newGMM)
    oldLogLik= numpy.sum(calculatedLogDensity)
    newLogLik= None
    newCalculatedLogDensity=None
    newGmm =None
    NMu = None
    NSeg = None
    NW = None
    
    if Type == "EM":
        NMu,NSeg,NW = EM(S,X)
    elif Type == "EM_Diag":
        NMu,NSeg,NW = EM_Diag(S,X)
    else :
        NMu,NSeg,NW = EM_Tied(S,X) 
    
    while True :
        newGmm =[[] for i in range(NMu.shape[1])]
        for i in range(NMu.shape[1]):
            newGmm[i].append(NW[i])
            newGmm[i].append(mcol(NMu[:,i]))
            newGmm[i].append(numpy.array(NSeg[i]))
       
        newCalculatedLogDensity, S= logpdf_GMM(X,newGmm)
        newLogLik = numpy.sum(newCalculatedLogDensity)
        
        if Type == "EM":
            NMu,NSeg,NW = EM(S,X)
        elif Type == "EM_Diag":
            NMu,NSeg,NW = EM_Diag(S,X)
        else :
            NMu,NSeg,NW = EM_Tied(S,X) 
        
        if newLogLik-oldLogLik < 10**-6 : 
            break
      
        oldLogLik = newLogLik
        calculatedLogDensity= newCalculatedLogDensity
    
    return newGmm
    

class GMM:
    def __init__ (self,Type,n_repeat) :
        self.Type = Type
        self.n_repeat = n_repeat
        self.gmm0= None
        self.gmm1 = None
        
    def train(self, DTR, LTR):
        D0 = DTR[:, LTR==0]
        D1 = DTR[:, LTR==1]
        
        mu0 = mcol(D0.mean(1))
        C0 = MLCovariance(D0, mu0)
        self.gmm0 = [(1.0,mu0,C0)]
        
        mu1 = mcol(D1.mean(1))
        C1 = MLCovariance(D1, mu1)
        self.gmm1 = [(1.0,mu1,C1)]
        
        for i in range(self.n_repeat):
            self.gmm0 = splitGmm(D0,self.gmm0,self.Type)
            self.gmm1 = splitGmm(D1,self.gmm1,self.Type)

    def evaluate (self,DTE):
        ll0,_ = logpdf_GMM(DTE, self.gmm0)
        ll1,_ = logpdf_GMM(DTE, self.gmm1)
    
        llr = ll1-ll0
        Plabel = [1 if ll1[i]>ll0[i] else 0 for i in range(len(ll0))]
        Plabel = numpy.array(Plabel)
        return Plabel, llr
        





