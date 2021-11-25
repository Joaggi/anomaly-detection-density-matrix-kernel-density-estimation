from load_arrhythmia import load_arrhythmia
from load_cardio import load_cardio
from load_kdd import load_kdd
from load_spambase import load_spambase
from load_thyroid import load_thyroid

def load_dataset(dataset):

    if(dataset == "arrhythmia"):
        return load_arrhythmia("data/arrhythmia.mat")
    if(dataset == "cardio"):
        return load_cardio("data/Cardiotocography.npy")
    if(dataset == "spambase"):
        return load_spambase("data/SpamBase.npy")
    if(dataset == "thyroid"):
        return load_thyroid("data/Thyroid.npy")
    if(dataset == "kddcup"):
        return load_kdd("data/kdd_cup.npz")