import pandas as pd
from data import select_features

def generate_submision_df(columns):
    df_test = pd.read_csv('../data/processed/test.csv',sep=",")
    X_sub = select_features(df_test)
    X_sub["EPRTRAnnexIMainActivityLabel_Chemical installations for the production on an industrial scale of basic organic chemicals: Organometallic compounds"] = 0
    X_sub["EPRTRAnnexIMainActivityLabel_Chemical installations for the production on an industrial scale of basic organic chemicals: Phosphorus-containing hydrocarbons"] = 0
    X_sub["EPRTRAnnexIMainActivityLabel_Industrial plants for the preservation of wood and wood products with chemicals"] = 0
    X_sub["EPRTRAnnexIMainActivityLabel_Installations for the building of, and painting or removal of paint from ships with a capacity for ships 100 m long"] = 0
    X_sub = X_sub[columns.values]
    
    return X_sub

def generate_submission_file(model, X, submission_number):
    df_test = pd.read_csv('../data/processed/test.csv', sep=",")
    y = model.predict(X)
    df_test["y_test"] = y
    df_sub = pd.DataFrame({'Id': df_test.index, 'pollutant': y})
    df_sub.to_csv(f"../submissions/submission_{submission_number}.csv", index=False)