################### Latest File ############################
import os
import cv2
import pickle
import random
import numpy as np
from PIL import Image
import math
import time
from scipy.special import rel_entr
from numpy.linalg import norm
from scipy.spatial import distance
from numpy import dot



def HIK3(arg):
    Xm, Yn, K = arg[0], arg[1], arg[2]
    submat = np.ndarray(shape=(len(Xm), len(Yn)), dtype=float)
    for i in range(len(Xm)):
        for j in range(len(Yn)):
            #submat[i][j] = np.true_divide(np.sum(np.minimum(Xm[i], Yn[j])), K)
            submat[i][j] = np.sum(np.minimum(Xm[i], Yn[j]))
     m1=np.min(submat)
     m2=np.max(submat)
    return [submat, m1, m2]

    
if __name__ == "__main__":
    startTime = time.time()
    os.chdir(r"/home/codefiles")
    from Step_2_Extract39DMFCC import compute39DMFCCforTestSignal
    from git_ExtractGaussianPosteriorgrams import restoreGMM, getPOSTFilesForWAVFiles
    

    os.chdir(r"/home/fundlab15/manisha/otherPickleFiles")
    picklefile = open('testKeywordsDictionary', 'rb')
    testKeywordsDictionary = pickle.load(picklefile)
    picklefile.close()
    picklefile = open('classDivisionOfTestFilesByKeywords', 'rb')
    classDivisionOfTestFilesByKeywords = pickle.load(picklefile)
    picklefile.close()
   
    g = restoreGMM(gmm_name = 'g64', path = r'/home/GMMfiles')
    n11 = 0
    n12 = 0
    n21 = 0
    n22 = 0
    #alpha=7
    
    keywords = ['artists', 'beautiful', 'carry', 'breakdown', 'greasy', 'development', 'wash', 'hostages', 'children', 'like that', 'dark suit', 'lunch', 'money', 'oily rag', 'popularity', 'problem', 'organizations', 'review', 'water', 'warm', 'woolen']
    for i in range(len(keywords)):
        sampleIndex = [0,1,2,3,4,5,6]
        #sampleIndex2 = [0,1,2,3,4,5,6]
        keywordFiles = [testKeywordsDictionary[keywords[i]][si] for si in sampleIndex]
        '''
        if len(testKeywordsDictionary[keywords[i]]) > 7:
            sampleIndex = random.sample(range(len(testKeywordsDictionary[keywords[i]])), 7)
            keywordFiles = [testKeywordsDictionary[keywords[i]][si] for si in sampleIndex]
        else:
            keywordFiles = testKeywordsDictionary[keywords[i]]
        '''
        sampleIndex2 = [q for q in range (0,len(classDivisionOfTestFilesByKeywords[i][0]))]
        classDivisionOfTestFilesByKeywords[i][1] = [classDivisionOfTestFilesByKeywords[i][1][si] for si in sampleIndex2]
        
        if len(classDivisionOfTestFilesByKeywords[i][1]) != 0:
            classDivisionOfTestFilesByKeywords[i][1] = getPOSTFilesForWAVFiles(classDivisionOfTestFilesByKeywords[i][1], name = 'g64')
        if len(classDivisionOfTestFilesByKeywords[i][0]) != 0:
            classDivisionOfTestFilesByKeywords[i][0] = getPOSTFilesForWAVFiles(classDivisionOfTestFilesByKeywords[i][0], name = 'g64')
        
        for keywordFile in keywordFiles:
            mfcc = compute39DMFCCforTestSignal(keywordFile.split('/')[-1], r"/home/fundlab15/manisha/implementation/Testing2")
            X =  g.predict_proba(mfcc )        
            K = len(X[0])
            xl = len(X)
            for j in range(len(classDivisionOfTestFilesByKeywords[i][0])):
                path, f = os.path.split(classDivisionOfTestFilesByKeywords[i][0][j])
                os.chdir(path)
                Y = np.genfromtxt(f)
                yl = len(Y)
                dtw = []
                #for HIK
                dtw =HIK3([X,Y,K])
                ndtw = dtw[0]
                l = dtw[1]
                h = dtw[2]
                hml = 255 / (h - l)
                ndtw = (ndtw - l) * hml
                img = np.zeros((xl, yl, 3), 'uint8')
                img[:, :, 0] = ndtw
                img[:, :, 1] = ndtw
                img[:, :, 2] = ndtw
                
                img = Image.fromarray(img, mode='RGB')
                img = img.resize((128,32), Image.ANTIALIAS)
                
                pathToSaveFiles = r"images/Train/class1"
                if j <= int(0.8 * len(classDivisionOfTestFilesByKeywords[i][0])):
                    os.chdir(pathToSaveFiles)
                    n11 += 1
                    img.save(str(n11) + '.bmp')
                    img.close()
                else:
                    os.chdir(r"images/Train/class1")
                    n21 += 1
                    img.save(str(n21) + '.bmp')
                    img.close()
                
            for j in range(len(classDivisionOfTestFilesByKeywords[i][1])):
                path, f = os.path.split(classDivisionOfTestFilesByKeywords[i][1][j])
                os.chdir(path)
                Y = np.genfromtxt(f)
                yl = len(Y)     
                dtw = []
                dtw =HIK3([X, Y, K])
                ndtw = dtw[0]
                l = dtw[1]
                h = dtw[2]
                
                hml = 255 / (h - l)
                ndtw = (ndtw - l) * hml
                img = np.zeros((xl, yl, 3), 'uint8')
                img[:, :, 0] = ndtw
                img[:, :, 1] = ndtw
                img[:, :, 2] = ndtw
                
                img = Image.fromarray(img, mode='RGB')
                img = img.resize((128,32), Image.ANTIALIAS)                
                pathToSaveFiles = r"/images/Train/class2"
                if j <= int(0.8 * len(classDivisionOfTestFilesByKeywords[i][1])):
                    os.chdir(pathToSaveFiles)
                    n12 += 1
                    img.save(str(n12) + '.bmp')
                    img.close()
                else:
                    os.chdir(r"/images/Test/class2")
                    n22 += 1
                    img.save(str(n22) + '.bmp')
                    img.close()    
               
    endTime = time.time()
    print("Time taken : ", endTime - startTime)
