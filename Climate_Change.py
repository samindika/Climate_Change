import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_ClimateChange= pd.read_csv("Climate_Change.csv",skiprows=4)
df_new_ClimateChange=df_ClimateChange[(df_ClimateChange["Indicator Name"]=="Urban population") | (df_ClimateChange["Indicator Name"]=="Agricultural land (sq. km)") | (df_ClimateChange["Indicator Name"]=="CO2 emissions (kt)") ].reset_index(drop=True)
df_new_country=df_new_ClimateChange.transpose()
df_new_country.columns = df_new_country.iloc[0]
df_new_country=df_new_country.iloc[0:,36:75]
df_new_country = df_new_country.dropna()


df_new_year=df_new_ClimateChange.iloc[36:72, :4].join(df_new_ClimateChange.iloc[36:75, 34:60]).reset_index(drop=True)
df_new_year = df_new_year.dropna()


df_new_year_Antigua=df_new_country["Antigua and Barbuda"]
df_new_year_Antigua=df_new_year_Antigua.drop(["Country Name", "Country Code", "Indicator Code"])
column_names = df_new_year_Antigua.iloc[0]
df_new_year_Antigua = df_new_year_Antigua[1:] #take the data less the header row
df_new_year_Antigua.columns = column_names
df_new_year_Antigua = df_new_year_Antigua.astype({'Urban population':'float','CO2 emissions (kt)':'float','Agricultural land (sq. km)':'float', })
#df_new_year_Antigua.info()
df_new_year_Antigua.describe()

df_new_year_Australia=df_new_country["Australia"]
df_new_year_Australia=df_new_year_Australia.drop(["Country Name", "Country Code", "Indicator Code"])
column_names = df_new_year_Australia.iloc[0]
df_new_year_Australia = df_new_year_Australia[1:] #take the data less the header row
df_new_year_Australia.columns = column_names
df_new_year_Australia = df_new_year_Australia.astype({'Urban population':'float','CO2 emissions (kt)':'float','Agricultural land (sq. km)':'float', })
#df_new_year_Antigua.info()
df_new_year_Australia.describe()

df_new_year_Bahrain=df_new_country["Bahrain"]
df_new_year_Bahrain=df_new_year_Bahrain.drop(["Country Name", "Country Code", "Indicator Code"])
column_names = df_new_year_Bahrain.iloc[0]
df_new_year_Bahrain = df_new_year_Bahrain[1:] #take the data less the header row
df_new_year_Bahrain.columns = column_names
df_new_year_Bahrain = df_new_year_Bahrain.astype({'Urban population':'float','CO2 emissions (kt)':'float','Agricultural land (sq. km)':'float', })
#df_new_year_Antigua.info()
df_new_year_Bahrain.describe()


Antigua_mode=df_new_year_Antigua["Urban population"].mode()[0]
Antigua_mode_mask=df_new_year_Antigua["Urban population"] == Antigua_mode
df_new_year_Antigua.loc[Antigua_mode_mask].iloc[:,0].to_frame()

Australia_mode=df_new_year_Australia["Urban population"].mode()[0]
Australia_mode_mask=df_new_year_Australia["Urban population"] == Australia_mode
df_new_year_Australia.loc[Australia_mode_mask].iloc[:,0].to_frame()

Bahrain_mode=df_new_year_Bahrain["Urban population"].mode()[0]
Bahrain_mode_mask=df_new_year_Bahrain["Urban population"] == Bahrain_mode
df_new_year_Bahrain.loc[Bahrain_mode_mask].iloc[:,0].to_frame()

column = [['Urban population','sum'],['Urban population','median'],['Urban population','mode'],['Co2 Emmisions','s1'],['Co2 Emmisions','m1'],['Co2 Emmisions','m2']]
statistics = {
    "sum": {
        "Antigua and Baradua": df_new_year_Antigua["Urban population"].sum(),
        "Australia": df_new_year_Australia["Urban population"].sum(),
        "Bahrain":df_new_year_Bahrain["Urban population"].sum()
    },
    "median": {
        "Antigua and Baradua": df_new_year_Antigua["Urban population"].median(),
        "Australia":df_new_year_Australia["Urban population"].median(),
        "Bahrain":df_new_year_Bahrain["Urban population"].median()
    },
    "mode": {
        "Antigua and Baradua": Antigua_mode,
        "Australia":Australia_mode,
        "Bahrain":Bahrain_mode
    },
    "s1": {
        "Antigua and Baradua": df_new_year_Antigua["CO2 emissions (kt)"].sum(),
        "Australia": df_new_year_Australia["CO2 emissions (kt)"].sum(),
        "Bahrain":df_new_year_Bahrain["CO2 emissions (kt)"].sum()
    },
    "m1": {
        "Antigua and Baradua": df_new_year_Antigua["CO2 emissions (kt)"].median(),
        "Australia":df_new_year_Australia["CO2 emissions (kt)"].median(),
        "Bahrain":df_new_year_Bahrain["CO2 emissions (kt)"].median()
    },
    "m2": {
        "Antigua and Baradua": Antigua_mode,
        "Australia":Australia_mode,
        "Bahrain":Bahrain_mode
    }
    

}

df_statistics = pd.DataFrame(data=statistics)
df_statistics.columns = pd.MultiIndex.from_tuples(column)
print(df_statistics)
