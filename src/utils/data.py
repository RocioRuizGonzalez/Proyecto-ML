import pandas as pd

def prepare_train_data(df):
    df.drop("Unnamed: 0",axis=1,inplace=True)
    df.drop("CONTINENT",axis=1,inplace=True)
    df.drop("targetRelease",axis=1,inplace=True)
    df['pollutant'].replace(['Nitrogen oxides (NOX)','Carbon dioxide (CO2)',"Methane (CH4)"],[0,1,2], inplace=True)
    if not df['pollutant'].isin([0,1,2]).all():
        raise Exception("pollutant contains invalid values")
    (x,y) = df.shape
    if y != 19:
        raise Exception("Dataframe contains incorrect number of columns")
    return df

def prepare_test_data(df):
    df.drop("CONTINENT",axis=1,inplace=True)
    df.drop("targetRelease",axis=1,inplace=True)
    (x,y) = df.shape
    if y != 18:
        raise Exception("Dataframe contains incorrect number of columns")
    return df

def select_features(df):
    X = df[["eprtrSectorName", 'EPRTRAnnexIMainActivityLabel', "avg_wind_speed", 'avg_temp', 'min_temp', 'min_wind_speed', 'max_temp', 'max_wind_speed', 'countryName', "DAY WITH FOGS"]]
    X = pd.get_dummies(X)
    return X