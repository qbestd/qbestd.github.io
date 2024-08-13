#Step 3


import os
import numpy as np
from sklearn import mixture
import pickle


#Build a GMM with components = 'no_of_components'
def buildGMM(F,no_of_components, iterations = 100):
    g = mixture.GaussianMixture(n_components=no_of_components, init_params='kmeans', max_iter = iterations)
    g.fit(F)
    return g


#Save the GMM for future usage
def saveGMM(g, gmm_name, path):
    os.chdir(path)
    np.savetxt(gmm_name + '_weights.txt', g.weights_)
    np.savetxt(gmm_name + '_means.txt', g.means_)
    np.save(gmm_name + '_covariances', g.covariances_)
    picklefile = open(gmm_name + '_gmm_pickle', 'ab')
    pickle.dump(g, picklefile) #Source, destination                    
    picklefile.close()
    

#Restore GMM for using
def restoreGMM(gmm_name, path = r'/GMMFiles'):
    os.chdir(path)
    picklefile = open(gmm_name + '_gmm_pickle', 'rb')      
    g = pickle.load(picklefile) 
    picklefile.close()
    return g


#Calculate Posteriorgrams for training examples with reduced features alpha-post_reduced for alpha=3, post_reduced3 for alpha=3
def calculatePosteriorgrams(g, filenames, gmm_name):
    i = 0   
    for f in filenames:
        p = os.path.split(f)
        os.chdir(p[0])
        feat = np.genfromtxt(p[1])
        post_feat = g.predict_proba(feat)       
        np.savetxt(gmm_name + p[1][:-4] + 'post', post_feat)
        if i%500 == 0:
            print(i)
        i = i + 1

def buildAndSaveGMM(F,k,iterations = 100, name = 'gmm', path = r'/GMMFiles'):
    g = buildGMM(F,k, iterations)
    while g.converged_ == False:
        print("Failed to converge in ", iterations, " iterations! Retrying with ", iterations + 100, " iterations")
        iterations += 100
        g = buildGMM(k, iterations)
    print("Your GMM has converged!")
    g.get_params()
    saveGMM(g, name, path)
    print('GMM saved successfully at following location: ' + path)
    
    
def getMFCCFilesList(path):
    filenames = []
    for roots,dirs,files in os.walk(path):
        for name in files:
            if name.endswith('.mfcc'):
                filenames.append(os.path.join(roots, name))
    return filenames


def stackMFCCs(filenames):
    F = []
    for f in filenames:
        p = os.path.split(f)
        os.chdir(p[0])
        feat = np.genfromtxt(p[1])
        #feat.resize(l)
        F.append(feat)
    F = np.vstack(F)
    return F


#Calculate Posterior grams for files
def calculatePosteriorgramsForGMM(name = 'gmm', directory = r'/TIMIT'):
    g = restoreGMM(name, path = r'GMMFiles')
    filenames = getMFCCFilesList(directory)
    calculatePosteriorgrams(g, filenames, name)
    
    
def getPOSTFilesForWAVFiles(filesList, name = 'gmm'):
    for i in range(len(filesList)):
        a = filesList[i].split('/')
        filesList[i] = '/'.join(a[:-1]+[name + a[-1].split('.')[0] + '.post'])
    return filesList

