import pickle

with open('filename.pkl', 'rb') as f:
    clf = pickle.load(f)