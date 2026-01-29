import numpy as np 

# def get_entropy(vector, delta=1e-8):
#     '''
#     Gets the shannon entropy of a vector whose elements are probabilities. Ignores zeros.
#     i.e. vector.sum() = 1

#     Even with less than ideal machine accuracy, entropy contribution of v --> 0^+ approaches zero.  
#     '''
#     if vector.isnull().values.all(): # if the vector has no contributions, then return np.nan
#         return np.nan 

#     v = vector[vector > delta].values
#     entropy = -sum(v * np.log(v))
    
#     return entropy

def get_entropy(vector, delta=1e-8):
    if sum(vector)==0: # if the vector has no contributions, then return np.nan
        return np.nan 

    v = vector[vector > delta]
    entropy = -sum(v * np.log(v))
    
    return entropy
