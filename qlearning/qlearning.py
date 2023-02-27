import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math
import time

class model:
    
    def __init__(self,S:list,A:list,gamma:float,Q:pd.DataFrame,alpha:float):
        self.S=S
        self.A=A
        self.gamma=gamma
        self.Q=Q
        self.alpha=alpha

def update(m:model, s, a, r, sp):
    s=np.array(s) #[1 to 100]
    a=np.array(a) #[1 to 4]
    sp=np.array(sp) #[1 to 100]
    
    s-=1
    a-=1
    sp-=1

    m.Q.values[s, a] += 0.01 * (r + m.gamma * m.Q.values[sp, :].max(axis = 1) - m.Q.values[s, a])
    return m

def read_csv(filename):
    data = pd.read_csv(filename)
    return data

def write_policy(m:model=None,states=None,actionoffset=None):
    index=list(m.Q.index)
    
    with open('large.policy','w') as file:
             df=m.Q.loc[states,:].idxmax(axis=1)
             csv=df.to_csv('large.policy',index=False)
    
def main():
    data=read_csv('data/small.csv') #index =[0,...,99999], columns=[s,a,r,sp],  states are in range [1,50000]
    
    all_possible_states = list(range(1, 100+1))
    all_possible_actions = list(range(1,4+1))
    
    # get the set difference between the reference list and the DataFrame index
    unvisited_states = list(set(all_possible_states) - set(data.iloc[:,0]))

    #all observed state indices
    S = list(pd.unique(data.loc[:,'s']))#length = 21960 , Note: 21960 + 28040 = visited states + unvisited states = 50000
    
    #all observed action indices
    A = list(pd.unique(data.loc[:,'a']))#[0 - 6]
    
    gamma = 0.95
    
    #store q(s,a) for all s,a
    Q = pd.DataFrame(
        np.zeros(  (len(all_possible_states),  len(all_possible_actions))  ), index=range(1,100+1),columns=range(1,4+1)
    )
      
    alpha = 0.01

    my_model = model(S,A,gamma,Q,alpha)
    
    t0 = time.time()
    
    #q learning
    num_iters=1000
    for j in range(num_iters):      
            print(f'CURRENT ITERATION: {j}')
            #data = shuffle(data)
            s =  data.loc[:, 's']
            a =  data.loc[:, 'a']
            r =  data.loc[:, 'r']
            sp = data.loc[:, 'sp']
            

            my_model = update(my_model, s, a, r, sp)
            
    t1 = time.time()
    print(f'RUNTIME WAS : {t1-t0} SECONDS')
    write_policy(my_model,states=all_possible_states)


    print(f'MY MODEL: {my_model.Q}')
    S_index0 = np.array(S)-1
    state_coords = np.unravel_index(S_index0,(10,10)) #[[x1,x2,...] ,[y1,y2,...] ]


    sx = state_coords[0] #[x1, x2, ...]
    sy = state_coords[1] #[y1, y2, ...]

    empty = pd.DataFrame(np.zeros(  (10,10)  ),index=range(10))

    
    for i in range(len(sx)):
        xc=sx[i]
        yc=sy[i]
        empty.iloc[xc,yc] = my_model.Q.loc[S_index0[i]+1,:].idxmax()
            
    sns.heatmap(empty,cmap='viridis',annot=True)
    plt.show()

if __name__ == "__main__":
    main()
    
   
