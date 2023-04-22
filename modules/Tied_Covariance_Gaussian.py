import numpy 
import math

def mcol(lst):
    return lst.reshape((lst.shape[0],1))

def mrow (lst):
    return lst.reshape((1,lst.shape[0]))

def MLCovariance (X,mu):
    mu = mcol(mu)
    CCov = numpy.dot((X-mu), (X-mu).T)
    CCov /= X.shape[1]
    return CCov

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


class Tied_Cov:
    def __init__(self,prior):
        self.D=[]
        self.muC = []
        self.cov = []
        self.TotCov = []
        self.difClasses = []
        self.S=[]
        self.Sjoint=[]
        self.prior = prior
        
    
    def train(self, DTR, LTR):
        self.D=[]
        self.difClasses = numpy.unique(LTR)
        for clsno in self.difClasses:
            self.D.append(DTR[:,LTR == clsno])
            self.muC.append(self.D[clsno].mean(1))
            if len(self.cov):
                self.cov[0] += self.D[clsno].shape[1] * MLCovariance(self.D[clsno],self.muC[clsno])
            else:
                self.cov.append(self.D[clsno].shape[1] * MLCovariance(self.D[clsno],self.muC[clsno]))
            
        self.TotCov = self.cov[0]/DTR.shape[1]
            
    def evaluate(self, DTE):
        self.S=[]
        self.Sjoint=[]
        if len(self.muC)!=2 :
            self.muC[0] = (self.muC[0]+ self.muC[2])/2
            self.muC[1] = (self.muC[1]+ self.muC[3])/2
            self.muC.pop()
            self.muC.pop()
            
        
        for i in self.difClasses:
            Y = logpdf_GAU_ND(DTE,mcol(self.muC[i]),self.TotCov)
            Y = numpy.exp(Y)
            self.S.append(Y)
        
        for i in range(len(self.S)):
            malePrior= self.prior
            femalePrior= 1- malePrior
            chPrior = malePrior if i==0 else femalePrior
            self.Sjoint.append(self.S[i]*chPrior)        

        SJoint= numpy.zeros((len(self.Sjoint),self.Sjoint[0].shape[0]))

        i=0
        for r in self.Sjoint:
            SJoint[i]= r
            i+=1
        SMarginal=mrow(SJoint.sum(0))
        SPost = SJoint/SMarginal
        llr = logLikelihoodRatio(self.S,self.prior)
        
        PLabels=numpy.argmax(SPost, axis=0)
        return PLabels,llr
