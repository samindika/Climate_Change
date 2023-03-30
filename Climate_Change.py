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

column_stat = [['Sum','Urban population'],['Sum','CO2 emissions (kt)'],['Sum','Agriculture'],['Median','Urban population1'],['Median','CO2 emissions (kt)1'],['Median','Agriculture1'],['Mode','Urban population2'],['Mode','CO2 emissions (kt)2'],['Mode','Agriculture2']]
statistics = {
    "Urban population": {
        "Bangladesh": df_new_year_Bangladesh["Urban population"].sum(),
        "Australia": df_new_year_Australia["Urban population"].sum(),
        "Bahrain":df_new_year_Bahrain["Urban population"].sum()
    },
    "CO2 emissions (kt)": {
        "Bangladesh": df_new_year_Bangladesh["CO2 emissions (kt)"].sum(),
        "Australia": df_new_year_Australia["CO2 emissions (kt)"].sum(),
        "Bahrain":df_new_year_Bahrain["CO2 emissions (kt)"].sum()
    },
    "Agriculture": {
        "Bangladesh": df_new_year_Bangladesh["Agricultural land (sq. km)"].sum(),
        "Australia": df_new_year_Australia["Agricultural land (sq. km)"].sum(),
        "Bahrain":df_new_year_Bahrain["Agricultural land (sq. km)"].sum()
    },
    
    
    "Urban population1": {
        "Bangladesh": df_new_year_Bangladesh["Urban population"].median(),
        "Australia": df_new_year_Australia["Urban population"].median(),
        "Bahrain":df_new_year_Bahrain["Urban population"].median()
    },
    "CO2 emissions (kt)1": {
        "Bangladesh": df_new_year_Bangladesh["CO2 emissions (kt)"].median(),
        "Australia": df_new_year_Australia["CO2 emissions (kt)"].median(),
        "Bahrain":df_new_year_Bahrain["CO2 emissions (kt)"].median()
    },
    "Agriculture1": {
        "Bangladesh": df_new_year_Bangladesh["Agricultural land (sq. km)"].median(),
        "Australia": df_new_year_Australia["Agricultural land (sq. km)"].median(),
        "Bahrain":df_new_year_Bahrain["Agricultural land (sq. km)"].median()
    },
    
    
    "Urban population2": {
        "Bangladesh": df_new_year_Bangladesh["Urban population"].median(),
        "Australia": df_new_year_Australia["Urban population"].median(),
        "Bahrain":df_new_year_Bahrain["Urban population"].median()
    },
    "CO2 emissions (kt)2": {
        "Bangladesh": df_new_year_Bangladesh["CO2 emissions (kt)"].median(),
        "Australia": df_new_year_Australia["CO2 emissions (kt)"].median(),
        "Bahrain":df_new_year_Bahrain["CO2 emissions (kt)"].median()
    },
    "Agriculture2": {
        "Bangladesh": df_new_year_Bangladesh["Agricultural land (sq. km)"].median(),
        "Australia": df_new_year_Australia["Agricultural land (sq. km)"].median(),
        "Bahrain":df_new_year_Bahrain["Agricultural land (sq. km)"].median()
    }
    
}

df_statistics = pd.DataFrame(data=statistics)
df_statistics.columns = pd.MultiIndex.from_tuples(column_stat)

Bangladesh_population_co2_corr = df_new_year_Bangladesh['Urban population'].corr(df_new_year_Bangladesh['CO2 emissions (kt)'], method='pearson')
Bangladesh_population_agriculture_corr = df_new_year_Bangladesh['Urban population'].corr(df_new_year_Bangladesh['Agricultural land (sq. km)'], method='pearson')
Bangladesh_co2_agriculture_corr = df_new_year_Bangladesh['CO2 emissions (kt)'].corr(df_new_year_Bangladesh['Agricultural land (sq. km)'], method='pearson')

Australia_population_co2_corr = df_new_year_Australia['Urban population'].corr(df_new_year_Australia['CO2 emissions (kt)'], method='pearson')
Australia_population_agriculture_corr = df_new_year_Australia['Urban population'].corr(df_new_year_Australia['Agricultural land (sq. km)'], method='pearson')
Australia_co2_agriculture_corr = df_new_year_Australia['CO2 emissions (kt)'].corr(df_new_year_Australia['Agricultural land (sq. km)'], method='pearson')

Bahrain_population_co2_corr = df_new_year_Bahrain['Urban population'].corr(df_new_year_Bahrain['CO2 emissions (kt)'], method='pearson')
Bahrain_population_agriculture_corr = df_new_year_Bahrain['Urban population'].corr(df_new_year_Bahrain['Agricultural land (sq. km)'], method='pearson')
Bahrain_co2_agriculture_corr = df_new_year_Bahrain['CO2 emissions (kt)'].corr(df_new_year_Bahrain['Agricultural land (sq. km)'], method='pearson')

column_cor = [['Bangladesh','Urban population'],['Australia','Urban population1'],['Bahrain','Urban population2']]
corelations = {
    "Urban population": {
        "CO2 emissions (kt)": Bangladesh_population_co2_corr,
        "Agricultural land (sq. km)": Bangladesh_population_agriculture_corr
        
    },
    "Urban population1": {
        "CO2 emissions (kt)": Australia_population_co2_corr,
        "Agricultural land (sq. km)": Australia_population_agriculture_corr
    },
    "rban population2": {
         "CO2 emissions (kt)": Bahrain_population_co2_corr,
        "Agricultural land (sq. km)": Bahrain_population_agriculture_corr
    }}

df_corelations = pd.DataFrame(data=corelations)
df_corelations.columns = pd.MultiIndex.from_tuples(column_cor)

