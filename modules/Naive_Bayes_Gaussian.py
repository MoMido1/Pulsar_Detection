import numpy 
import math

def mcol(lst):
    return lst.reshape((lst.shape[0],1))

def mrow (lst):
    return lst.reshape((1,lst.shape[0]))

def ML_Diag_Covariance (X,mu):
    mu = mcol(mu)
    CCov = numpy.dot((X-mu), (X-mu).T)
    CCov /= X.shape[1]
    de = CCov.diagonal()
    DiagCov = numpy.zeros((CCov.shape[0], CCov.shape[1]))
    numpy.fill_diagonal(DiagCov, de)

    return DiagCov

def logLikelihoodRatio (S,prior):
    malePrior= prior
    femalePrior = 1- prior
    Sjoint = []
    for i in range(len(S)):
        chPrior = malePrior if i==0 else femalePrior
        Sjoint.append(S[i]*chPrior)  
    epsilon = 1e-5
    llr= numpy.log (Sjoint[1]+epsilon)-numpy.log(Sjoint[0]+epsilon)
    return llr

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

class Naive_Bayes:
    def __init__(self,prior):
        self.D = []
        self.muC = []
        self.cov = []
        self.difClasses = []
        self.S = []
        self.Sjoint =[]
        self.prior = prior
        
    
    def train(self, DTR, LTR):
        self.D=[]
        self.difClasses = numpy.unique(LTR)
        for clsno in self.difClasses:
            self.D.append(DTR[:,LTR == clsno])
            self.muC.append(self.D[clsno].mean(1))
            self.cov.append(ML_Diag_Covariance(self.D[clsno],self.muC[clsno]))
            
            
    def evaluate(self, DTE):
        self.S=[]
        self.Sjoint=[]
        
        if len(self.muC)!=2 :
            self.muC[0] = (self.muC[0]+ self.muC[2])/2
            self.muC[1] = (self.muC[1]+ self.muC[3])/2
            self.cov[0] = (self.cov[0]+ self.cov[2])/2
            self.cov[1] = (self.cov[1]+ self.cov[3])/2
            self.muC.pop()
            self.muC.pop()
            self.cov.pop()
            self.cov.pop()
        
        for clsno in self.difClasses:
            Y = logpdf_GAU_ND(DTE,mcol(self.muC[clsno]),self.cov[clsno])

            Y = numpy.exp(Y)
            self.S.append(Y)

        
        for i in range(len(self.S)):
            malePrior= self.prior
            femalePrior= 1- malePrior
            chPrior = malePrior if i==0 else femalePrior
            self.Sjoint.append(self.S[i]*chPrior)    

        self.SJoint= numpy.zeros((len(self.Sjoint),self.Sjoint[0].shape[0]))

        i=0
        for r in self.Sjoint:
            self.SJoint[i]= r
            i+=1

        SMarginal=mrow(self.SJoint.sum(0))
        SPost = self.SJoint/SMarginal
        llr = logLikelihoodRatio(self.S,self.prior)
        
        PLabels=numpy.argmax(SPost, axis=0)
        return PLabels,llr
    
