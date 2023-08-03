import pandas as pd
import numpy as np
from debm import Prospect, PAS, makeGrid, resPlot, FitMultiGame, saveEstimation
from debm.Models import PAS, Hard_PASs
import csv

# Do not forget to install DEBM: https://github.com/ofiryakobi/debm
df = pd.read_excel("target problems2023.07.27.xlsx", sheet_name="Sheet1").replace('.', None)
df['a2'] = df['a2'].fillna(df['a1'])

df['rbc'] = [0 for i in range(len(df))]
df['rab'] = df['corr']

# try to play with this
df['sa'] = [0 for _ in range(len(df))]
df['sb'] = [0 for _ in range(len(df))]
# Set up
trials = 100  # All decision tasks are 100 trials
sims = 1000  # Simulate x times when making predictions
k_calc = 10  # Sample size
delta_calc = 0.2  # delta


# Define the Prospect function
def Prospect_func_6c(uniformMatrix, a1, pa1, a2, corr, Generate: bool, AB_prospects=None, sd=0):
    # This is a helper function to accomodate all parameters of the choice problems
    # corr: 0,1,-1; Generate: Regenerate uniformMatrix prospect; AB_prospects: for add parameter
    if Generate:
        uniformMatrix.Generate()
    if corr == 1 or corr == 0:
        shared_outcomes = uniformMatrix.outcomes.copy()
    elif corr == -1:
        shared_outcomes = 1 - uniformMatrix.outcomes.copy()
    outcomes = np.full(shared_outcomes.shape, a2)
    outcomes[np.where(shared_outcomes < pa1)] = a1
    if AB_prospects != None:
        outcomes += (AB_prospects[0].outcomes + AB_prospects[1].outcomes) / 2
    if sd != 0:
        outcomes += np.random.normal(0, sd, shared_outcomes.shape)
    return outcomes


# Read all info from the file and create a matrix of all prospects (each line is a problem)
prospect_matrix = []
problem_code=[]
a0rates=[]
for index, row in df.iterrows():
    prospects = []
    problem_code.append(row['prob'])
    a0rates.append(row['Est_A0'])
    a1, a2 = row['a1'], row['a2']
    pa1, pa2 = row['pa1'], 1 - row['pa1']
    b1, b2 = row['b1'], row['b2']
    pb1, pb2 = row['pb1'], 1 - row['pb1']
    rab, rbc = row['rab'], row['rbc']
    sa, sb= row['sa'], row['sb']
    uniformProspect = Prospect(trials, np.random.uniform, False,
                               size=trials)  # This is the shared uniform distribution to sample from
    prospects.append(Prospect(trials, Prospect_func_6c, False, uniformProspect, a1=a1, pa1=pa1, a2=a2, corr=1,
                              Generate=True, AB_prospects=None, sd=sa))  # A
    prospects.append(Prospect(trials, Prospect_func_6c, False, uniformProspect, a1=b1, pa1=pb1, a2=b2, corr=rab,
                              Generate=row['rab'] == 0, AB_prospects=None, sd=sb))  # B
    prospect_matrix.append(prospects)
# Generate the predictions using PAS, print and save to file
predictions = []  # this will store the predictions for all the problems
for i, problem in enumerate(prospect_matrix):
    pas_6c = Hard_PASs({'Kappa': k_calc, 'Omega': 0, 'Delta': delta_calc, 'A0rate': a0rates[i]}, problem, sims)  # Set the parameters from the paper
    print(f"Problem {problem_code[i]}: P(a)={df['pa1'][i]} a1={df['a1'][i]} a2={df['a2'][i]} P(b)={df['pb1'][i]} b1={df['b1'][i]} b2={df['b2'][i]}")
    pred,Arates= pas_6c.Predict()
    AAAbest, AAAnotbest, ABAbest, ABAnotbest = np.round(Arates, 3)
    Arate1, Arate2, Arate3, Arate4 = np.round(np.mean(np.split(pred, 4), axis=1), 3)[:, 0]
    predictions.append([problem_code[i],Arate1, Arate2, Arate3, Arate4, AAAbest, AAAnotbest, ABAbest, ABAnotbest])
    print(f'P: ' + str(predictions[-1])) # print the predictions
pred_df=pd.DataFrame(predictions,columns=['prob','Arate1', 'Arate2', 'Arate3', 'Arate4','AAAbest','AAAnotb','ABAbest','ABAnotb'])
pred_df.to_csv('predictions_target.csv',index=False)
