#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:36:26 2023

@author: fundlab15
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:07:17 2023

@author: fundlab15
"""



import os
import pickle
import multiprocessing as mp
import numpy as np
from PIL import Image
import time

def HIK7(X, Ypath, K, pathToSaveFiles, n):
    
    path, f = os.path.split(Ypath)
    os.chdir(path)
    Y = np.genfromtxt(f)
    dtw = np.ndarray(shape=(len(X), len(Y)), dtype=float)
    l = 9999
    h = -9999
    xl = len(X)
    yl = len(Y)
    for i in range(xl):
        for j in range(yl):
            dtw[i][j] = np.true_divide(np.sum(np.minimum(X[i], Y[j])), K)
            if dtw[i][j] < l:
                l = dtw[i][j]
            if dtw[i][j] > h:
                h = dtw[i][j]
    hml = 255 / (h - l)
    ndtw = (dtw - l) * hml
    img = np.zeros((xl, yl, 3), 'uint8')
    img[:, :, 0] = ndtw
    img[:, :, 1] = ndtw
    img[:, :, 2] = ndtw
    
    img = Image.fromarray(img, mode='RGB')
    img = img.resize((128, 32), Image.ANTIALIAS)#
        
    os.chdir(pathToSaveFiles)
    img.save(str(n) + '.bmp')
    img.close()
   
if __name__ == "__main__":
    startTime = time.time()
    
    os.chdir(r"/codefiles")
    from Step_2_Extract39DMFCC import compute39DMFCCforTestSignal
    from Step_3_ExtractGaussianPosteriorgrams import restoreGMM, getPOSTFilesForWAVFiles
     
    picklefile = open('testKeywordsDictionary', 'rb')
    testKeywordsDictionary = pickle.load(picklefile)
    picklefile.close()
    picklefile = open('classDivisionOfTestFilesByKeywords', 'rb')
    classDivisionOfTestFilesByKeywords = pickle.load(picklefile)
    picklefile.close()
    
    g = restoreGMM(gmm_name = 'g32', path = r'/home/fundlab15/manisha/implementation')
    n11 = 0
    n12 = 0
    n21 = 0
    n22 = 0
    
    pool = mp.Pool(processes = 6) 
    
    keywords = ['artists', 'beautiful', 'carry', 'breakdown', 'greasy', 'development', 'wash', 'hostages', 'children', 'like that', 'dark suit', 'lunch', 'money', 'oily rag', 'popularity', 'problem', 'organizations', 'review', 'water', 'warm', 'woolen']
    for i in range(len(keywords)):
        sampleIndex = [0,1,2,3,4,5,6]
        #sampleIndex2 = [0,1,2,3,4,5,6]
        keywordFiles = [testKeywordsDictionary[keywords[i]][si] for si in sampleIndex]
        #keywordFiles = keywordsFilesList[i]
        sampleIndex2 = [q for q in range (0,len(classDivisionOfTestFilesByKeywords[i][0]))]
        #sampleIndex2 = random.sample(range(len(classDivisionOfTestFilesByKeywords[i][1])), len(classDivisionOfTestFilesByKeywords[i][0]))
        classDivisionOfTestFilesByKeywords[i][1] = [classDivisionOfTestFilesByKeywords[i][1][si] for si in sampleIndex2]
        
        if len(classDivisionOfTestFilesByKeywords[i][1]) != 0:
            classDivisionOfTestFilesByKeywords[i][1] = getPOSTFilesForWAVFiles(classDivisionOfTestFilesByKeywords[i][1], name = 'g32')
        if len(classDivisionOfTestFilesByKeywords[i][0]) != 0:
            classDivisionOfTestFilesByKeywords[i][0] = getPOSTFilesForWAVFiles(classDivisionOfTestFilesByKeywords[i][0], name = 'g32')
       
        
        for keywordFile in keywordFiles:
            #mfcc = compute39DMFCCforTestSignal(keywordFile.split('\\')[-1], r"E:\GitHub\Spoken Term Detection\Spoken-Term-Detection\Implementation\Testing2")
            mfcc = compute39DMFCCforTestSignal(keywordFile.split('/')[-1], r"Testing2")
            
            X = g.predict_proba(mfcc)    
            K = len(X[0])
                        
            for j in range(len(classDivisionOfTestFilesByKeywords[i][0])):
                 if j <= int(0.8 * len(classDivisionOfTestFilesByKeywords[i][0])):
                    pathToSaveFiles1 = r"Train/class1"
                    n11 += 1
                    res = [pool.apply_async(HIK7, (X, classDivisionOfTestFilesByKeywords[i][0][j], K, pathToSaveFiles1, n11))]
                 else:
                    pathToSaveFiles2 = r"Test/class1"
                    n21 += 1
                    res=[pool.apply_async(HIK7, (X, classDivisionOfTestFilesByKeywords[i][0][j], K, pathToSaveFiles2, n21))]
                     
                      
            for j in range(len(classDivisionOfTestFilesByKeywords[i][1])):
                if j <= int(0.8 * len(classDivisionOfTestFilesByKeywords[i][1])):
                    pathToSaveFiles1 = r"Train/class2"
                    #os.chdir(pathToSaveFiles)
                    n12 += 1
                    res = [pool.apply_async(HIK7, (X, classDivisionOfTestFilesByKeywords[i][1][j], K, pathToSaveFiles1, n12))]
                else:
                    pathToSaveFiles2 = r"Test/class2"
                    n22 += 1
                    res=[pool.apply_async(HIK7, (X, classDivisionOfTestFilesByKeywords[i][1][j], K, pathToSaveFiles2, n22))]
            
           
    pool.close()
    pool.join()
    endTime = time.time()
    print("Time taken : ", endTime - startTime)