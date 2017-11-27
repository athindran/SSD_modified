import pickle

relevantvideos = pickle.load(open('relevantvideos.p','rb'))
print(relevantvideos[0])
