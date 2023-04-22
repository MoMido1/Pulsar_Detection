import numpy
import scipy.optimize
from numpy.linalg import norm


def logreg_obj (v,*args):
    w,b = v[0:-1] , v[-1]
    dtr= args[0]
    ltr = args[1]
    lmda= args[2]
    prior= args[3]
    n = ltr.shape[0]
    sumation=0
    
    dtp=dtr[:,ltr==1] 
    dtn=dtr[:,ltr==0]
    
    logexpp= numpy.logaddexp(0,-(numpy.dot(w.T,dtp)+b))
    logexpn= numpy.logaddexp(0,(numpy.dot(w.T,dtn)+b))
    
    sumationp = numpy.sum(logexpp)
    sumationn = numpy.sum(logexpn)
    
    J = lmda/2 * norm(w)**2 + prior/dtp.shape[1] * (sumationp) + (1-prior)/dtn.shape[1]* sumationn
    return J


class LR :
    def __init__(self,prior,l):
        self.Wopt = []
        self.bopt = 0
        self.l = l
        self.prior=prior
    
    def train(self, DTR, LTR):
        self.LTR = LTR
        x0 =numpy.zeros(DTR.shape[0]+1)
        mybounds = [(-1000,1000) for _ in range(DTR.shape[0]+1)]
        x,f,d=scipy.optimize.fmin_l_bfgs_b(logreg_obj,x0,fprime=None,args=(DTR,LTR,self.l,self.prior),approx_grad=True,iprint=1,bounds= mybounds)
        self.Wopt= numpy.array(x[0:-1])
        self.bopt = x[-1]     
            
    def evaluate(self, DTE):
        predictedCls=numpy.zeros(DTE.shape[1],dtype=int)
        SS=numpy.zeros(DTE.shape[1],dtype=int)
        if len(self.Wopt)==0:
            pass
        else:
            SS= numpy.dot(self.Wopt,DTE)+self.bopt + numpy.log(self.prior / (1-self.prior))
            predictedCls = SS>0
           
        return predictedCls,SS
    
    
    
    