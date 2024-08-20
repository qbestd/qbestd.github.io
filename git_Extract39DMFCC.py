#Step 2


import os
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import delta
import numpy as np


#Compute 39 Dimensional MFCC features for training signals
def computeAndSave39DMFCC(path=r'TIMIT'):
    filenames = []
    for roots,dirs,files in os.walk(path):
        for name in files:
            if name.endswith('.wav'):
                filenames.append(os.path.join(roots, name))
    print(len(filenames))
    
    for f in filenames:
        p = os.path.split(f)
        os.chdir(p[0])
        (rate,sig) = wav.read(p[1])
        mfcc_feat = mfcc(sig,rate)
        d_mfcc_feat = delta(mfcc_feat, 2)
        d_d_mfcc_feat = delta(d_mfcc_feat, 2)
        feat = np.hstack([mfcc_feat, d_mfcc_feat, d_d_mfcc_feat])
        np.savetxt(p[1][:-3]+'mfcc', feat)
        #a = np.genfromtxt(p[1][:-3]+'mfcc')
    print("Successfully extracted 39 Dimensional MFCC features")
    
    
#Compute 39 Dimensional MFCC features for a test signal
def compute39DMFCCforTestSignal(file_name, path=r'/TIMIT/TEST'):
    os.chdir(path)
    (rate,sig) = wav.read(file_name)
    mfcc_feat = mfcc(sig,rate,nfft=1200)
    d_mfcc_feat = delta(mfcc_feat, 2)
    d_d_mfcc_feat = delta(d_mfcc_feat, 2)
    feat = np.hstack([mfcc_feat, d_mfcc_feat, d_d_mfcc_feat])
    return feat

