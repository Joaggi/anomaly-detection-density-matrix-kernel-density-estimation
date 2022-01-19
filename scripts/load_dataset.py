from load_arrhythmia import load_arrhythmia
from load_cardio import load_cardio
from load_kdd import load_kdd
from load_spambase import load_spambase
from load_thyroid import load_thyroid
from load_glass import load_glass

def load_dataset(dataset, setting):

    if(dataset == "arrhythmia"):
        return load_arrhythmia("data/arrhythmia.mat", setting)
    if(dataset == "cardio"):
        return load_cardio("data/Cardiotocography.npy", setting)
    if(dataset == "spambase"):
        return load_spambase("data/SpamBase.npy", setting)
    if(dataset == "thyroid"):
        return load_thyroid("data/Thyroid.npy", setting)
    if(dataset == "kddcup"):
        return load_kdd("data/kdd_cup.npz", setting)
    if(dataset == "glass"):
        return load_glass("data/glass.mat", setting)