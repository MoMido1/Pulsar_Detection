import matplotlib.pyplot as plt
import numpy 
from modules.Multivariate_Gaussian import MVG
from modules.Naive_Bayes_Gaussian import Naive_Bayes
from modules.Tied_Covariance_Gaussian import Tied_Cov
from modules.Tied_Covariance_NB_Gaussian import Tied_NB
from modules.Logistic_Regression import LR
from modules.Linear_SVM import L_SVM
from modules.Kernel_SVM import K_SVM
from modules.GMM_Class import GMM
from Library.lib import * 
import seaborn as sb
from scipy.stats import norm



########################## Loading Datasets ##########################################
DTR, LTR = load('data/Train.txt')
DTE, LTE = load('data/Test.txt')



def features_Analysis(DTR,LTR,DTE):
    L0_DTR = DTR[:,LTR==0] #negative class
    L1_DTR = DTR[:,LTR==1] #positive class
    
    print("The number of Training samples for the class Negative are "+str(L0_DTR.shape[1]))
    print("The number of Training samples for the class Positive are "+str(L1_DTR.shape[1]))
    
    print("The number of Features are "+str(DTR.shape[0]))
    
    for i in range(DTR.shape[0]):
        
        plt.figure('features Distribution')
        plt.xlabel("Feature"+str(i+1))
        plt.ylabel("Contibution")
        plt.hist(L0_DTR[i,:],bins = 40,density=True,label='Negative',edgecolor='b',alpha=0.65)
        plt.hist(L1_DTR[i,:],bins = 40,density=True,label='Positive',edgecolor='r',alpha=0.65)
        plt.legend(loc='upper right')
        plt.show()   
   
    # Gaussianize the Training and Testing features 
    
    y = numpy.zeros(DTR.shape)
    
    for i in range(DTR.shape[1]) :
        x = DTR[:,i]
        count = numpy.sum(x.reshape((x.shape[0],1))<DTR,axis=1)
        rank = (count+1)/ (DTR.shape[1]+2)
        y[:,i]= norm.ppf(rank)
    z = numpy.zeros(DTE.shape)
    
    for i in range(DTR.shape[1]) :
        x = DTE[:,i]
        count = numpy.sum(x.reshape((x.shape[0],1))<DTE,axis=1)
        rank = (count+1)/ (DTE.shape[1]+2)
        z[:,i]= norm.ppf(rank)
    
    L0_ = y[:,LTR==0] #negative class
    L1_ = y[:,LTR==1] #positive class
    
    DTE = z
    for i in range(DTR.shape[0]):
        
        plt.figure('features Distribution')
        plt.xlabel("Feature"+str(i+1))
        plt.ylabel("Contibution")
        plt.hist(L0_[i,:],bins = 40,density=True,label='Negative',edgecolor='b',alpha=0.65)
        plt.hist(L1_[i,:],bins = 40,density=True,label='Positive',edgecolor='r',alpha=0.65)
        plt.legend(loc='upper right')
        plt.show()   
   
    # Displaying heatmap
    allDataset = abs(numpy.corrcoef(DTR))
    sb.heatmap(allDataset, cmap="Greys", annot=False)
    plt.xlabel("Total Data Features Correlation")
    plt.ylabel("Raw Data")
    
    plt.show()
    posSamples = abs(numpy.corrcoef(L1_DTR))
    sb.heatmap(posSamples, cmap="Blues", annot=False)
    plt.xlabel("Positive Class Freatures Correlation")
    plt.ylabel("Raw Data")
    plt.show()
    negSamples = abs(numpy.corrcoef(L0_DTR))
    sb.heatmap(negSamples, cmap="Reds", annot=False)
    plt.xlabel("Negative Class Freatures Correlation")
    plt.ylabel("Raw Data")
    plt.show()
    
    # displaying heatmap after Gaussianization
   
    allDataset = abs(numpy.corrcoef(y))
    sb.heatmap(allDataset, cmap="Greys", annot=False)
    plt.xlabel("Total Data Freatures Correlation")
    plt.ylabel("Gaussianized Data")
    plt.show()
    posSamples = abs(numpy.corrcoef(L1_))
    sb.heatmap(posSamples, cmap="Blues", annot=False)
    plt.xlabel("Positive Class Freatures Correlation")
    plt.ylabel("Gaussianized Data")
    plt.show()
    negSamples = abs(numpy.corrcoef(L0_))
    sb.heatmap(negSamples, cmap="Reds", annot=False)
    plt.xlabel("Negative Class Freatures Correlation")
    plt.ylabel("Gaussianized Data")
    plt.show()
    
    DTR=y
    return DTR,DTE


def singleSplit(model,prior,DTR):
    (Dtr, Ltr),(Dte, Lte) = split_db_2to1(DTR,LTR)
    model.train(Dtr, Ltr)
    results,llr = model.evaluate(Dte)
    accuracy = Compute_Accuracy(Lte, results)
    print("Evaluation Accuracy = ", round(accuracy, 1)," %")
    minDcf= Min_DCF(llr,Lte,prior)
    print("minDCF for single split evaluation is "+str(minDcf))
    

def AllTests (DTR):
    priors = [0.1,0.5,0.9]
    print("\n############## Multivariate Gaussian #################")
    for i in priors:   
        print("\n\nApplication with prior of "+str(i))
        print("------------------------------------")
        print("Single Split")
        print("--------------------------\n")
        mmvg = MVG(i)
        singleSplit(mmvg ,i,DTR)
        print("\nK-fold")
        print("-------------------------\n")
        mvg = MVG(i)
        evaluation_acc,t = Kfold(mvg, DTR, LTR, i, K = 5)
   
    print("\n############## Naive Bayes Gaussian ##################")
    for i in priors:   
        print("\n\nApplication with prior of "+str(i))
        print("------------------------------------")
        print("Single Split")
        print("--------------------------\n")
        NB = Naive_Bayes(i)
        singleSplit(NB , i,DTR)
        print("\nK-fold")
        print("-------------------------\n")
        NB_1 = Naive_Bayes(i)
        evaluation_acc,t = Kfold(NB_1, DTR, LTR, i, K = 5)
   

    print("\n############## Tied Covariance Gaussian #################")
    for i in priors:   
        print("\n\nApplication with prior of "+str(i))
        print("------------------------------------")
        print("Single Split")
        print("--------------------------\n")
        TC = Tied_Cov(i)
        singleSplit(TC , i,DTR)
        print("\nK-fold")
        print("-------------------------\n")
        TC_1 = Tied_Cov(i)
        evaluation_acc,t = Kfold(TC_1, DTR, LTR, i, K = 5)
       

    print("\n######### Tied Covariance & Naive Bayes Gaussian ############")
    for i in priors:   
        print("\n\nApplication with prior of "+str(i))
        print("------------------------------------")
        print("Single Split")
        print("--------------------------\n")
        TNB = Tied_NB(i)
        singleSplit(TNB , i,DTR)
        print("\nK-fold")
        print("-------------------------\n")
        TNB_1 = Tied_NB(i)
        evaluation_acc,t = Kfold(TNB_1, DTR, LTR, i, K = 5)
       
        
    print("\n######### Logistic Regression ############")
    lmd = numpy.array([0.00001,0.0001,0.001,0.01,0.1,1,10])
    tot_minDCF = numpy.zeros((len(priors),len(lmd)),dtype= numpy.float32 )
    k=0
    for i in priors:   
        print("\n\nApplication with prior of "+str(i))
        print("------------------------------------")
        print("K-fold")
        print("-------------------------\n")
        
        for j in range(lmd.shape[0]):
            lr = LR(i,l=lmd[j])
            print("Lambda : "+str(lmd[j]))
            print("-------------------------")
            evaluation_acc,t = Kfold(lr, DTR, LTR, i, K = 5)
            
            tot_minDCF[k,j] = t
        k+=1
    plt.plot(lmd, tot_minDCF[0], label='prior of 0.1')
    plt.plot(lmd, tot_minDCF[1], label='prior of 0.5')
    plt.plot(lmd, tot_minDCF[2], label='prior of 0.9')
    plt.legend(loc="upper left")
    plt.xlabel('Lamda')
    plt.ylabel('minDCF LR')
    plt.xscale('log')
    plt.show()
    
    print("\n######### Linear Support Vector Machine ############")
    c = numpy.array([0.0001,0.001,0.01,0.1,1]) 
    tot_minDCF = numpy.zeros((len(priors),len(c)),dtype= numpy.float32 )
    k=0
    for i in priors:   
        print("\n\nApplication with prior of "+str(i))
        print("------------------------------------")
        print("K-fold")
        print("-------------------------")
        
        for j in range(c.shape[0]):
            print("C : "+str(c[j])+", K : 1")
            print("-------------------------\n")
            lsvm = L_SVM(1,c[j],i)
            evaluation_acc,t = Kfold(lsvm, DTR, LTR, i, K = 5)
            tot_minDCF[k,j] = t
        
        k+=1        
    plt.plot(c, tot_minDCF[0], label='prior of 0.1')
    plt.plot(c, tot_minDCF[1], label='prior of 0.5')
    plt.plot(c, tot_minDCF[2], label='prior of 0.9')
    plt.legend(loc="upper left")
    plt.xlabel('C')
    plt.ylabel("minDCF Linear SVM")
    plt.xscale('log')
    plt.show()
    

    print("\n######### Kernel Support Vector Machine ############")
    tot_minDCF = numpy.zeros((len(priors),4),dtype= numpy.float32 )
    lmda = numpy.array([1,10])
    c = numpy.array([0,1])
    k=0
    for i in priors:   
        print("\n\nApplication with prior of "+str(i))
        print("------------------------------------")
        print("K-fold")
        print("-------------------------\n")
        print("Performance of RBF kernel")
       
        for j in range(lmda.shape[0]):
            print("Lamda : "+str(lmda[j])+" , K :0 , C:1")
            print("-------------------------\n")
            #setting the K =0 C = 1
            k_svm =K_SVM ('RBF',0,1,0,0,lmda[j],i)
            evaluation_acc,t = Kfold(k_svm, DTR, LTR, i, K = 5)
            tot_minDCF[k,j]=t
    
        print("-------------------------\n")
        print("Performance of Poly kernel")
        
        for j in range(c.shape[0]):
            print("c : "+str(c[j])+" , K:1 , C:1 , d :2")
            print("-------------------------\n")
            #setting the K =0 C = 1
            k_svm =K_SVM ('Poly',1,1,2,c[j],0,i)
            evaluation_acc,t = Kfold(k_svm, DTR, LTR, i, K = 5)
            tot_minDCF[k,j+2]=t
        k+=1
           
    plt.plot(lmda, tot_minDCF[0][0:2], label='prior of 0.1')
    plt.plot(lmda, tot_minDCF[1][0:2], label='prior of 0.5')
    plt.plot(lmda, tot_minDCF[2][0:2], label='prior of 0.9')
    plt.legend(loc="upper left")
    plt.ylabel('minDCF RBF Kernel SVM')
    plt.xlabel('Lamda')
    plt.xscale('log')
    plt.figure()
    plt.plot(c, tot_minDCF[0][2:], label='prior of 0.1')
    plt.plot(c, tot_minDCF[1][2:], label='prior of 0.5')
    plt.plot(c, tot_minDCF[2][2:], label='prior of 0.9')
    plt.legend(loc="upper left")
    plt.ylabel('minDCF Poly Kernel SVM')
    plt.xlabel('c')
    plt.xscale('log')
    plt.show()
    
        
    print("\n############## GMM #################")
    
    for i in priors:   
        print("\n\nApplication with prior of "+str(i))
        print("------------------------------------")
        print("Single Split")
        print("--------------------------\n")
        
        print("Using the EM GMM")
        print("--------------------------")
        gmm_mod = GMM('EM',3)
        singleSplit(gmm_mod , i,DTR)
        print("\nK-fold")
        print("-------------------------\n")
        gmm_mod_ = GMM('EM',3)
        evaluation_acc,t = Kfold(gmm_mod_, DTR, LTR, i, K = 5)
        
       
        print("\nUsing the EM Diagonal GMM")
        print("--------------------------")
        gmm_mod = GMM('EM_Diag',3)
        singleSplit(gmm_mod , i,DTR)
        print("\nK-fold")
        print("-------------------------\n")
        gmm_mod_ = GMM('EM_Diag',3)
        evaluation_acc,t = Kfold(gmm_mod_, DTR, LTR, i, K = 5)
        
        print("\nUsing the EM Tied GMM")
        print("--------------------------")
        gmm_mod = GMM('EM_Tied',3)
        singleSplit(gmm_mod , i,DTR)
        print("\nK-fold")
        print("-------------------------\n")
        gmm_mod_ = GMM('EM_Tied',3)
        evaluation_acc,t = Kfold(gmm_mod_, DTR, LTR, i, K = 5)
        
            

#------------------- main----------------------------#

# Feature analysis and Gaussianizing the training and Testing set
DTR_,DTE_ = features_Analysis(DTR,LTR,DTE)

############################# Trying Raw Data #################################
print("\n-------------- Without Gaussianization ------------------") 
AllTests(DTR)

print("\n--- Gaussianized Without Any Dimensionality Reduction ---")
AllTests(DTR_)

print("\n------------- Gaussianized with PCA = 4 ---------------")
DTR=PCA(DTR_,4)[0]
AllTests(DTR)

print("\n------------- Gaussianized with PCA = 7 ---------------")
DTR=PCA(DTR_,7)[0]
AllTests(DTR)

plt.show()


#*****************************************************************************#
#************ Score Calipration and actual testing on unknown data************#

def act_minDCF_plot (llr,LTR) :
    effPriorLogOdds = numpy.linspace(-3,3,21)
    effPrior= 1/(1+numpy.exp(-effPriorLogOdds))

    dcfNorm = numpy.zeros(effPriorLogOdds.shape)
    minNormdcf=numpy.zeros(effPriorLogOdds.shape)

    for j in range(effPrior.shape[0]):
        dcfNorm[j]=Act_DCF(llr,LTR,effPrior[j])
        minNormdcf[j]=Min_DCF(llr,LTR,effPrior[j])
    
    plt.plot(effPriorLogOdds, dcfNorm, label='act DCF', color='r')
    plt.plot(effPriorLogOdds, minNormdcf, label='min DCF', color='b')
    plt.legend(loc='upper right')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()


# MVG Tied Covariance PCA=4 ,prior =0.5  actual and min DCF
DTR=PCA(DTR_,4)[0]
mmvg = Tied_Cov(0.5)
mmvg.train(DTR, LTR)
results,llr0 = mmvg.evaluate(DTR)
plt.figure()
plt.xlabel("prior")
plt.ylabel("MVG Tied")
act_minDCF_plot(llr0, LTR)

# Logistic Regression PCA=4 ,prior =0.5 ,lambda=10^-5 actual and min DCF
DTR=PCA(DTR_,4)[0]
lr = LR(0.5,l=0.00001)
lr.train(DTR, LTR)
results,llr1 = lr.evaluate(DTR)
plt.figure()
plt.xlabel("prior")
plt.ylabel("Logistic Regression")
act_minDCF_plot(llr1, LTR)

# Linear SVM PCA=4 ,prior =0.5 ,C=1 actual and min DCF
DTR=PCA(DTR_,4)[0]
lsvm = L_SVM(1,1,0.5)
lsvm.train(DTR, LTR)
results,llr2 = lsvm.evaluate(DTR)
plt.figure()
plt.xlabel("prior")
plt.ylabel("Linear SVM")
act_minDCF_plot(llr2, LTR)

# GMM PCA=4 ,prior =0.5 ,Full Covariance actual and min DCF
DTR=PCA(DTR_,4)[0]
gmm_ = GMM('EM',3)
gmm_.train(DTR, LTR)
results,llr3 = gmm_.evaluate(DTR)
plt.figure()
plt.xlabel("prior")
plt.ylabel("GMM Full Covariance")
act_minDCF_plot(llr3, LTR)


#Calibrating and plotting the change
lrr = LR(0.5,l=1e-4)
llr_0,a0,b0= calibrate_scores(lrr,llr0,LTR,0.5)
llr_0=llr_0[0]
plt.figure()
plt.xlabel("prior")
plt.ylabel("Calibrated MVG Tied")
act_minDCF_plot(llr_0, LTR)


llr_1,a1,b1= calibrate_scores(lrr,llr1,LTR,0.5)
llr_1=llr_1[0]
plt.figure()
plt.xlabel("prior")
plt.ylabel("Calibrated Logistic Regression")
act_minDCF_plot(llr_1, LTR)

llr_2,a2,b2= calibrate_scores(lrr,llr2,LTR,0.5)
llr_2=llr_2[0]
plt.figure()
plt.xlabel("prior")
plt.ylabel("Calibrated Linear SVM")
act_minDCF_plot(llr_2, LTR)


llr_3,a3,b3= calibrate_scores(lrr,llr3,LTR,0.5)
llr_3=llr_3[0]
plt.figure()
plt.xlabel("prior")
plt.ylabel("Calibrated GMM Full covariance")
act_minDCF_plot(llr_3, LTR)

#Testing
#Tied Cov

results,llrt = mmvg.evaluate(DTE)
scores = a0 * llrt + b0 - numpy.log(0.5/(1 - 0.5))
mn_dcf=Min_DCF(scores,LTE,0.5)
ac_dcf= Act_DCF(scores,LTE,0.5)
print("Tied Covarinace Final Testing \n")
print("-------------------------")
print("min DCF is "+ str(mn_dcf))
print("actual DCF is "+ str(ac_dcf))
plb= scores>0
acc=Compute_Accuracy(LTE,plb)
print("Total Accuracy is "+str(acc))


#Logistic Regression Testing

results,llrt = lr.evaluate(DTE)
scores = a1 * llrt + b1 - numpy.log(0.5/(1 - 0.5))
mn_dcf=Min_DCF(scores,LTE,0.5)
ac_dcf= Act_DCF(scores,LTE,0.5)
print("Logistic Regression Final Testing \n")
print("-------------------------")
print("min DCF is "+ mn_dcf)
print("actual DCF is "+ ac_dcf)
plb= scores>0
acc=Compute_Accuracy(LTE,plb)
print("Total Accuracy is "+acc)


#Linear SVM Testing

results,llrt = lsvm.evaluate(DTE)
scores = a2 * llrt + b2 - numpy.log(0.5/(1 - 0.5))
mn_dcf=Min_DCF(scores,LTE,0.5)
ac_dcf= Act_DCF(scores,LTE,0.5)
print("Linear SVM Final Testing \n")
print("-------------------------")
print("min DCF is "+ mn_dcf)
print("actual DCF is "+ ac_dcf)
plb= scores>0
acc=Compute_Accuracy(LTE,plb)
print("Total Accuracy is "+acc)


#GMM Testing

results,llrt = gmm_.evaluate(DTE)
scores = a3 * llrt + b3 - numpy.log(0.5/(1 - 0.5))
mn_dcf=Min_DCF(scores,LTE,0.5)
ac_dcf= Act_DCF(scores,LTE,0.5)
print("GMM Final Testing \n")
print("-------------------------")
print("min DCF is "+ mn_dcf)
print("actual DCF is "+ ac_dcf)
plb= scores>0
acc=Compute_Accuracy(LTE,plb)
print("Total Accuracy is "+acc)



