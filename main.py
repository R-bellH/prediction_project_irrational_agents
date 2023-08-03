import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor as dtg
from sklearn.model_selection import cross_val_score


def main():
    df1 = pd.read_excel("training101sum.xlsx", sheet_name="Sheet1").replace('.', float(0))
    df2=pd.read_csv("predictions_10.csv")

    for col in ['Arate1', 'Arate2', 'Arate3', 'Arate4','AAAbest','AAAnotb','ABAbest','ABAnotb']:
        y=df1[col]
        y_pred=df2[col]
        print(col)
        # calculate MAE, MSE, MSD
        # print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))
        # print('Mean Squared Deviation:', np.sqrt(metrics.mean_squared_error(y, y_pred)))

if __name__ == '__main__':
    # main()
    # df1 = pd.read_excel("training101sum.xlsx", sheet_name="Sheet1").replace('.', float(0))
    # df2 = pd.read_csv("predictions_10.csv")
    # # join on 'prob'
    # df3 = pd.merge(df1, df2, on='prob')
    # print(df3.columns)
    # print(df3)
    df = pd.read_excel("target problems2023.07.27.xlsx", sheet_name="Sheet1")
    df_pred = pd.read_csv("predictions_target.csv")
    for col in ['Arate1', 'Arate2', 'Arate3', 'Arate4', 'AAAbest', 'AAAnotb', 'ABAbest', 'ABAnotb']:
        df[col] = df_pred[col]
    print(df.columns)
    print(df)
    df.to_csv("Carmel Kenneth (211785670) Arbel Hadar(315775585).csv", index=False)




