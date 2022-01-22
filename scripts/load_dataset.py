from load_mat import load_mat_file
from load_cardio import load_cardio
from load_kdd import load_kdd
from load_spambase import load_spambase
from load_thyroid import load_thyroid


def load_dataset(dataset, setting):

    if(dataset == "arrhythmia"):
        return load_mat_file("data/arrhythmia.mat", setting)
    if(dataset == "cardio"):
        return load_cardio("data/Cardiotocography.npy", setting)
    if(dataset == "spambase"):
        return load_spambase("data/SpamBase.npy", setting)
    if(dataset == "thyroid"):
        return load_thyroid("data/Thyroid.npy", setting)
    if(dataset == "kddcup"):
        return load_kdd("data/kdd_cup.npz", setting)
    if(dataset == "glass"):
        return load_mat_file("data/glass.mat", setting)
    if(dataset == "ionosphere"):
        return load_mat_file("data/ionosphere.mat", setting)
    if(dataset == "letter"):
        return load_mat_file("data/letter.mat", setting)
    if(dataset == "mnist"):
        return load_mat_file("data/mnist.mat", setting)
    if(dataset == "musk"):
        return load_mat_file("data/musk.mat", setting)