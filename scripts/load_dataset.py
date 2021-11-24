from load_arrhythmia import load_arrhythmia

def load_dataset(dataset):

    if(dataset == "arrhythmia"):
        return load_arrhythmia("data/arrhythmia.mat")