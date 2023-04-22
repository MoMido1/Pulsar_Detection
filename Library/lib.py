import numpy
import scipy
import scipy.linalg
import math


def load(File):
    Data = []
    Labels = []
    with open(File) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = line.split(',')[-1]
                Data.append(attrs)
                Labels.append(label)
            except:
                pass    
    Data = numpy.hstack(Data)
    Labels = numpy.array(Labels, dtype=numpy.int32)
    return Data, Labels

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

def mcol (lst):
    return lst.reshape((lst.shape[0],1))

def mrow (lst):
    return lst.reshape((1,lst.shape[0]))

def unique(list1): 
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)       
    return sorted(unique_list)

def PCA (DataMat, m):
    mu = DataMat.mean(1)
    mu = mcol(mu)
    DataCentered = DataMat - mu
    Cov = numpy.dot(DataCentered,DataCentered.T)
    Cov = Cov / DataCentered.shape[1]
    s, U = numpy.linalg.eigh(Cov)
    P = U[:, ::-1][: , 0:m]
    DataProjected = numpy.dot(P.T,DataMat)
    return DataProjected ,P


def LDA (DataMat, ClassMat, m):
    mu =DataMat.mean(1)
    mu= mcol(mu)
    D=[]
    clsMeans=[]
    n=[]
    SB = []
     
    clss = unique(ClassMat)
    for i in clss:
        D.append(DataMat[:,ClassMat == i])
        clsMeans.append(mcol(D[i].mean(1)))
        n.append(D[i].shape[1])
        SB.append(numpy.dot((clsMeans[i]-mu),(clsMeans[i]-mu).T))

    tot_SB = numpy.zeros((SB[0].shape[0],SB[0].shape[1]))
    
    for i in clss:
        SB[i] = SB[i]*n[i]
        tot_SB += SB[i]    

    tot_SB = tot_SB/DataMat.shape[1]

    DC=[]
    SW = []
    
    for i in clss:
        DC.append(D[i]-clsMeans[i])
        SW.append(numpy.dot(DC[i],DC[i].T))
        
    tot_SW= sum(SW)
    tot_SW = tot_SW/DataMat.shape[1]
    
    s,U = scipy.linalg.eigh(tot_SB,tot_SW)

    W = U[:, ::-1][:,0:m]

    DataProjected = numpy.dot(W.T,DataMat)
    return DataProjected


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def Compute_Accuracy(Y, Y_predict):
    compare = Y_predict[Y_predict == Y]
    accuracy = compare.shape[0]/Y.shape[0] *100
    return accuracy

def Test_Model(model, X_train, Y_train, X_test, Y_test):
    model.train(X_train, Y_train)
    results,llr = model.evaluate(X_test)
    accuracy = Compute_Accuracy(Y_test, results)
    print("Testing Accuracy = ", round(accuracy, 1)," %")
    return accuracy,llr


def Kfold(model, X, Y, prior, K):
    n_attributes, n_samples = X.shape
    X = X.T
    n_samples_per_fold = int(n_samples/K)
    starting_index = 0
    ending_index = n_samples_per_fold
    total_accuracy = 0
    total_Dcf=2
    
    for i in range(K):
        # Compute the testing samples
        X_test = X[starting_index : ending_index]
        Y_test = Y[starting_index : ending_index]
        
        # Compute the training samples
        X_train_part1 = X[0 : starting_index]
        X_train_part2 = X[ending_index: -1]
        X_train = numpy.concatenate((X_train_part1, X_train_part2), axis = 0)
        
        Y_train_part1 = Y[0 : starting_index]
        Y_train_part2 = Y[ending_index: -1]
        Y_train = numpy.concatenate((Y_train_part1, Y_train_part2), axis = 0)
        
        # Apply to the model and get accuracy
        model.train(X_train.T, Y_train)

        
        results,llr = model.evaluate(X_test.T)
        total_accuracy += Compute_Accuracy(Y_test, results)
        try:
            min_dcf = Min_DCF(llr,Y_test,prior)
            total_Dcf =  min_dcf if min_dcf < total_Dcf else total_Dcf
        except:
            pass
        
        # Updating indexes for next iteration
        starting_index += n_samples_per_fold
        ending_index += n_samples_per_fold
        
    avg_accuracy = total_accuracy/K
    print("Evaluation Average Accuracy = ", round(avg_accuracy, 1)," %")
    
    # avg_Dcf = total_Dcf/K
    print("MinDCF for Kfold Evaluation = "+str(total_Dcf))
    
    return avg_accuracy,total_Dcf


def Threshold (p1,Cfn,Cfp):
    t = - math.log((p1*Cfn)/((1-p1)*Cfp))
    return t

def Threshold_1 (Cfn,Cfp):
    return math.log(Cfn/Cfp)
    
def DCF (p1,Cfn,Cfp,p00,p01,p10,p11):
    FNR = p01/(p01+p11)
    FPR = p10/(p00+p10)
    
    return p1*Cfn*FNR+(1-p1)*Cfp*FPR

def Bdummy (p1,Cfn,Cfp):
    par1= p1*Cfn
    par2= (1-p1)*Cfp
    
    return par1 if par1<par2 else par2


def Norm_DCF (DCF,Bmin):
    return DCF/Bmin

def Min_DCF(llr,labels,p1):
    Cfn=1
    Cfp=1
    sortedLLR= sorted(llr)
    T =[min(sortedLLR)-1, *sortedLLR, max(sortedLLR)+1]
    DCFall=2
    u=0
    
    for t in T:
        bayasPrediction_t= llr>t
        p00=0
        p01=0
        p10=0
        p11=0
        
        p00=numpy.sum((bayasPrediction_t == labels) & (bayasPrediction_t ==0))
        p10=numpy.sum((bayasPrediction_t != labels) & (bayasPrediction_t !=0))
        p01=numpy.sum((bayasPrediction_t != labels) & (bayasPrediction_t ==0))
        p11=numpy.sum((bayasPrediction_t == labels) & (bayasPrediction_t !=0))
        dcf = DCF(p1,Cfn,Cfp,p00,p01,p10,p11)
        
        if dcf<DCFall:
            DCFall= dcf
        u+=1     
    b=Bdummy (p1,Cfn,Cfp)   
    return Norm_DCF(DCFall, b)

def Act_DCF(llr,labels,p1):
    Cfn=1
    Cfp=1
    bayasPrediction_t= llr>Threshold(p1, Cfn, Cfp)
    
    p00=0
    p01=0
    p10=0
    p11=0
    
    p00=numpy.sum((bayasPrediction_t == labels) & (bayasPrediction_t ==0))
    p10=numpy.sum((bayasPrediction_t != labels) & (bayasPrediction_t !=0))
    p01=numpy.sum((bayasPrediction_t != labels) & (bayasPrediction_t ==0))
    p11=numpy.sum((bayasPrediction_t == labels) & (bayasPrediction_t !=0))
    
    b=Bdummy (p1,Cfn,Cfp)    
    dcf=DCF(p1,Cfn,Cfp,p00,p01,p10,p11)
    
    return Norm_DCF(dcf, b)


def calibrate_scores(model,llr, LTR,prior):
    llr= mrow(llr)
    model.train(llr, LTR)
    alpha = model.Wopt
    beta = model.bopt
    scores = alpha * llr + beta - numpy.log(prior/(1 - prior))
    return scores,alpha,beta

