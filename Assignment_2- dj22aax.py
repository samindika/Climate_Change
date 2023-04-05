"import modules"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def read_file(file_path):
    """
    Reads a CSV file containing climate change data and returns two pandas 
    dataframes:
    - df_new_country: a dataframe transposed by country, with country names as
    columns and years as rows.
    - df_new_year: a dataframe transposed by year, with countries as columns 
    and indicators as rows.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        two pandas dataframes (df_new_country, df_new_year)
    """

    df_ClimateChange = pd.read_csv(file_path, skiprows=4)

    # Filter pandas dataframe to include rows with specific indicators and reset the index
    df_ClimateChange = df_ClimateChange[(df_ClimateChange["Indicator Name"] == "Urban population") | (
        df_ClimateChange["Indicator Name"] == "Agricultural land (sq. km)") |
        (df_ClimateChange["Indicator Name"] == "CO2 emissions (kt)")].reset_index(drop=True)

    # Transpose by country
    df_new_country = df_ClimateChange.transpose()
    df_new_country.columns = df_new_country.iloc[0]
    df_new_country = df_new_country.iloc[0:, 36:75]
    df_new_country = df_new_country.dropna()

    # Transpose by year
    df_new_year = df_ClimateChange.iloc[36:72, :4].join(
        df_ClimateChange.iloc[36:72, 34:60]).reset_index(drop=True)
    df_new_year = df_new_year.dropna()

    # Return transposed dataframes
    return df_new_country, df_new_year


def retrieve_countrydata(df_new_country):
    """
    Retrieves the data for specific countries from a given DataFrame

    Parameters:
       df_new_country: A pandas DataFrame containing the transposed data of 
       climate change by country name

    Returns:
       DataFrames for four specific countries: Bangladesh, Australia, Bahrain, 
       and Burundi
   """
    # Retrive data for Country - Bangladesh
    df_new_year_Bangladesh = df_new_country["Bangladesh"]
    df_new_year_Bangladesh = df_new_year_Bangladesh.drop(
        ["Country Name", "Country Code", "Indicator Code"])
    column_names = df_new_year_Bangladesh.iloc[0]

    # take the data less the header row
    df_new_year_Bangladesh = df_new_year_Bangladesh[1:]
    df_new_year_Bangladesh.columns = column_names
    df_new_year_Bangladesh = df_new_year_Bangladesh.astype(
        {'Urban population': 'float', 'CO2 emissions (kt)': 'float',
         'Agricultural land (sq. km)': 'float', })

    # Retrive data for country - Australia
    df_new_year_Australia = df_new_country["Australia"]
    df_new_year_Australia = df_new_year_Australia.drop(
        ["Country Name", "Country Code", "Indicator Code"])
    column_names = df_new_year_Australia.iloc[0]

    # take the data less the header row
    df_new_year_Australia = df_new_year_Australia[1:]
    df_new_year_Australia.columns = column_names
    df_new_year_Australia = df_new_year_Australia.astype(
        {'Urban population': 'float', 'CO2 emissions (kt)': 'float',
         'Agricultural land (sq. km)': 'float', })
    df_new_year_Australia.describe()

    # Retrieve data for Bahrain
    df_new_year_Bahrain = df_new_country["Bahrain"]
    df_new_year_Bahrain = df_new_year_Bahrain.drop(
        ["Country Name", "Country Code", "Indicator Code"])
    column_names = df_new_year_Bahrain.iloc[0]

    # take the data less the header row
    df_new_year_Bahrain = df_new_year_Bahrain[1:]
    df_new_year_Bahrain.columns = column_names
    df_new_year_Bahrain = df_new_year_Bahrain.astype(
        {'Urban population': 'float', 'CO2 emissions (kt)': 'float',
         'Agricultural land (sq. km)': 'float', })

    # Retrive data for Burundi
    df_new_year_Burundi = df_new_country["Burundi"]
    df_new_year_Burundi = df_new_year_Burundi.drop(
        ["Country Name", "Country Code", "Indicator Code"])
    column_names = df_new_year_Burundi.iloc[0]

    # take the data less the header row
    df_new_year_Burundi = df_new_year_Burundi[1:]
    df_new_year_Burundi.columns = column_names
    df_new_year_Burundi = df_new_year_Burundi.astype(
        {'Urban population': 'float', 'CO2 emissions (kt)': 'float',
         'Agricultural land (sq. km)': 'float', })

    return df_new_year_Bangladesh, df_new_year_Australia, df_new_year_Bahrain, df_new_year_Burundi


def calculate_statistics(df_new_year_Bangladesh, df_new_year_Australia,
                         df_new_year_Bahrain, df_new_year_Burundi):
    """
    Calculates statistical data for the data sets of four countries - 
    Bangladesh, Australia, Bahrain, and Burundi
    The statistical data includes skewness, sum, median, and mode for three columns
    - Urban population, CO2 emissions (kt), and Agricultural land (sq. km).
    Finally, stores all statistical data in a dictionary called 'statistics'.

    Parameters:
        df_new_year_Bangladesh: Bangladesh dataframe
        df_new_year_Australia: Australia dataframe
        df_new_year_Bahrain: Bahrain dataframe
        df_new_year_Burundi: Burundi dataframe

    Returns:
        df_statistics dictionary 
    """

    # Mode Calculation - Bangladesh
    Bangladesh_Urban_mode = df_new_year_Bangladesh["Urban population"].mode()[
        0]
    Bangladesh_co2_mode = df_new_year_Bangladesh["CO2 emissions (kt)"].mode()[
        0]
    Bangladesh_agriculture_mode = df_new_year_Bangladesh["Agricultural land (sq. km)"].mode()[
        0]

    # Mode Calculation - Australia
    Australia_Urban_mode = df_new_year_Australia["Urban population"].mode()[0]
    Australia_co2_mode = df_new_year_Australia["CO2 emissions (kt)"].mode()[0]
    Australia_agriculture_mode = df_new_year_Australia["Agricultural land (sq. km)"].mode()[
        0]

    # Mode Calculation - Bahrain
    Bahrain_Urban_mode = df_new_year_Bahrain["Urban population"].mode()[0]
    Bahrain_co2_mode = df_new_year_Bahrain["CO2 emissions (kt)"].mode()[0]
    Bahrain_agriculture_mode = df_new_year_Bahrain["Agricultural land (sq. km)"].mode()[
        0]

    # Mode Calculation - Burundi
    Burundi_Urban_mode = df_new_year_Burundi["Urban population"].mode()[0]
    Burundi_co2_mode = df_new_year_Burundi["CO2 emissions (kt)"].mode()[0]
    Burundi_agriculture_mode = df_new_year_Burundi["Agricultural land (sq. km)"].mode()[
        0]

    """Present statistical data in a dictionary - Skewness, Sum, Median, Mode 
    for Bangladesh, Australia, Bahrain and Burundi"""

    column_stat = [['Skewness', 'Skew_Urban population'],
                   ['Skewness', 'Skew_CO2 emissions (kt)'],
                   ['Skewness', 'Skew_Agricultural land (sq. km)'],
                   ['Sum', 'Sum_Urban population'], [
                       'Sum', 'Sum_CO2 emissions (kt)'],
                   ['Sum', 'Sum_Agricultural land (sq. km)'],
                   ['Median', 'Med_Urban population'],
                   ['Median', 'Med_CO2 emissions (kt)'],
                   ['Median', 'Med_Agricultural land (sq. km)'],
                   ['Mode', 'Mod_Urban population'],
                   ['Mode', 'Mod_CO2 emissions (kt)'],
                   ['Mode', 'Mod_Agricultural land (sq. km)']]
    statistics = {
        "Skew_Urban population": {
            "Bangladesh": stats.skew(df_new_year_Bangladesh["Urban population"]),
            "Australia": stats.skew(df_new_year_Australia["Urban population"]),
            "Bahrain": stats.skew(df_new_year_Bahrain["Urban population"]),
            "Burundi": stats.skew(df_new_year_Burundi["Urban population"])
        },
        "Skew_CO2 emissions (kt)": {
            "Bangladesh": stats.skew(df_new_year_Bangladesh["CO2 emissions (kt)"]),
            "Australia": stats.skew(df_new_year_Australia["CO2 emissions (kt)"]),
            "Bahrain": stats.skew(df_new_year_Bahrain["CO2 emissions (kt)"]),
            "Burundi": stats.skew(df_new_year_Burundi["CO2 emissions (kt)"])
        },
        "Skew_Agricultural land (sq. km)": {
            "Bangladesh": stats.skew(df_new_year_Bangladesh["Agricultural land (sq. km)"]),
            "Australia": stats.skew(df_new_year_Australia["Agricultural land (sq. km)"]),
            "Bahrain": stats.skew(df_new_year_Bahrain["Agricultural land (sq. km)"]),
            "Burundi": stats.skew(df_new_year_Burundi["Agricultural land (sq. km)"])
        },

        "Sum_Urban population": {
            "Bangladesh": df_new_year_Bangladesh["Urban population"].sum(),
            "Australia": df_new_year_Australia["Urban population"].sum(),
            "Bahrain": df_new_year_Bahrain["Urban population"].sum(),
            "Burundi": df_new_year_Burundi["Urban population"].sum()
        },
        "Sum_CO2 emissions (kt)": {
            "Bangladesh": df_new_year_Bangladesh["CO2 emissions (kt)"].sum(),
            "Australia": df_new_year_Australia["CO2 emissions (kt)"].sum(),
            "Bahrain": df_new_year_Bahrain["CO2 emissions (kt)"].sum(),
            "Burundi": df_new_year_Burundi["CO2 emissions (kt)"].sum()
        },
        "Sum_Agricultural land (sq. km)": {
            "Bangladesh": df_new_year_Bangladesh["Agricultural land (sq. km)"].sum(),
            "Australia": df_new_year_Australia["Agricultural land (sq. km)"].sum(),
            "Bahrain": df_new_year_Bahrain["Agricultural land (sq. km)"].sum(),
            "Burundi": df_new_year_Burundi["Agricultural land (sq. km)"].sum()
        },

        "Med_Urban population": {
            "Bangladesh": df_new_year_Bangladesh["Urban population"].median(),
            "Australia": df_new_year_Australia["Urban population"].median(),
            "Bahrain": df_new_year_Bahrain["Urban population"].median(),
            "Burundi": df_new_year_Burundi["Urban population"].median()
        },
        "Med_CO2 emissions (kt)": {
            "Bangladesh": df_new_year_Bangladesh["CO2 emissions (kt)"].median(),
            "Australia": df_new_year_Australia["CO2 emissions (kt)"].median(),
            "Bahrain": df_new_year_Bahrain["CO2 emissions (kt)"].median(),
            "Burundi": df_new_year_Burundi["CO2 emissions (kt)"].median()
        },
        "Med_Agricultural land (sq. km)": {
            "Bangladesh": df_new_year_Bangladesh["Agricultural land (sq. km)"].median(),
            "Australia": df_new_year_Australia["Agricultural land (sq. km)"].median(),
            "Bahrain": df_new_year_Bahrain["Agricultural land (sq. km)"].median(),
            "Burundi": df_new_year_Burundi["Agricultural land (sq. km)"].median()
        },

        "Mod_Urban population": {
            "Bangladesh": Bangladesh_Urban_mode,
            "Australia": Australia_Urban_mode,
            "Bahrain": Bahrain_Urban_mode,
            "Burundi": Burundi_Urban_mode
        },
        "Mod_CO2 emissions (kt)": {
            "Bangladesh": Bangladesh_co2_mode,
            "Australia": Australia_co2_mode,
            "Bahrain": Bahrain_co2_mode,
            "Burundi": Burundi_co2_mode
        },
        "Mod_Agricultural land (sq. km)": {
            "Bangladesh": Bangladesh_agriculture_mode,
            "Australia": Australia_agriculture_mode,
            "Bahrain": Bahrain_agriculture_mode,
            "Burundi": Burundi_agriculture_mode
        }}

    df_statistics = pd.DataFrame(data=statistics)
    df_statistics.columns = pd.MultiIndex.from_tuples(column_stat)
    return df_statistics


def correlation_calculation(df_new_year_Bangladesh, df_new_year_Australia,
                            df_new_year_Bahrain, df_new_year_Burundi):
    """
  Calculates the correlation matrix for four different countries.

  Parameters:
      df_new_year_Bangladesh: DataFrame containing the data for Bangladesh
      df_new_year_Australia: DataFrame containing the data for Australia
      df_new_year_Bahrain: DataFrame containing the data for Bahrain
      df_new_year_Burundi: DataFrame containing the data for Burundi
  Returns:
      correlation matrix for each country
  """
    corr_bangladesh = df_new_year_Bangladesh.corr()
    corr_australia = df_new_year_Australia.corr()
    corr_bahrain = df_new_year_Bahrain.corr()
    corr_burundi = df_new_year_Burundi.corr()
    return corr_bangladesh, corr_australia, corr_bahrain, corr_burundi


def create_multiindex_columns(df_new_country):
    """
    Create a new DataFrame with a MultiIndex column structure
    Parameters:
        df_new_country : Dataframe transposed by country
    Returns:
        A new DataFrame with the same data as df_new_country, but with column 
        labels converted to a MultiIndex structure
    """

    df_new_country_multiindex = df_new_country.copy()
    new_columns = df_new_country_multiindex.iloc[[0, 2]].apply(
        lambda x: '_'.join(x.astype(str)), axis=0)

    # Set the columns as a MultiIndex
    df_new_country_multiindex.columns = pd.MultiIndex.from_product([
                                                                   new_columns])
    df_new_country_multiindex = df_new_country_multiindex.drop(
        df_new_country.index[:4])
    return df_new_country_multiindex


def plot_population_trends(df_new_country_multiindex):
    """
    Lineplot - Plots the urban population trends of different countries against time.

    Parameters:
    df_new_country_multiindex : DataFrame
        A pandas DataFrame with multi-index columns where the first level 
        represents the country name and the second level represents the population data 

    """
    plt.figure(figsize=(9, 6), facecolor="white")

    # plot the line graph
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Australia_Urban population"],
             label="Australia", linestyle=":", color="#e41a1c")
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Bangladesh_Urban population"],
             label="Bangladesh", linestyle=":", color="#377eb8")
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Bahrain_Urban population"],
             label="Bahrain", linestyle="-", color="#4daf4a")
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Benin_Urban population"],
             label="Benin", linestyle="-", color="#984ea3")
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Austria_Urban population"],
             label="Austria", linestyle="-", color="#ff7f00")
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Antigua and Barbuda_Urban population"],
             label="Antigua and Barabuda", linestyle="-", color="#ffff33")
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Bosnia and Herzegovina_Urban population"],
             label="Bosnia and Herzegovina", linestyle="-", color="#a65628")
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Bulgaria_Urban population"],
             label="Bulgaria", linestyle=":", color="#f781bf")

    # Labeling
    plt.xlabel("Year")
    plt.ylabel("Urban Population (in millions)")
    plt.xticks(rotation=45)

    # Chart Title
    plt.title("Urban Population Trends", fontweight='bold')
    plt.xlim(min(df_new_country_multiindex.index,),
             max(df_new_country_multiindex.index,))
    plt.ticklabel_format(axis='y', style='plain')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.savefig("population_linplot.png")
    plt.show()


def plot_co2emmision_overtime(df_new_country_multiindex):
    """
    Lineplot - Plot the CO2 emision trends of different countries against time
    Parameters:
        df_new_country_multiindex: A pandas multi-index dataframe

    """
    plt.figure(figsize=(9, 6), facecolor="white")

    # plot the line graph
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Australia_CO2 emissions (kt)"],
             label="Australia", linestyle=":", color="#e60049")
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Bangladesh_CO2 emissions (kt)"],
             label="Bangladesh", linestyle="-", color="#0bb4ff")
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Bahrain_CO2 emissions (kt)"],
             label="Bahrain", linestyle="-", color="#50e991")
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Benin_CO2 emissions (kt)"],
             label="Benin", linestyle="-", color="#e6d800")
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Austria_CO2 emissions (kt)"],
             label="Austria", linestyle=":", color="#9b19f5")
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Antigua and Barbuda_CO2 emissions (kt)"],
             label="Antigua and Barabuda", linestyle="-", color="#ffa300")
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Bosnia and Herzegovina_CO2 emissions (kt)"],
             label="Bosnia and Herzegovina", linestyle="-", color="#dc0ab4")
    plt.plot(df_new_country_multiindex.index,
             df_new_country_multiindex["Bulgaria_CO2 emissions (kt)"],
             label="Bulgaria", linestyle=":")

    # Labeling
    plt.xlabel("Year")
    plt.ylabel("CO2 Emissions (kilotons)")
    plt.xticks(rotation=45)

    # Chart Title
    plt.title("CO2 Emissions Trends", fontweight='bold')
    plt.xlim(min(df_new_country_multiindex.index,),
             max(df_new_country_multiindex.index,))
    plt.ticklabel_format(axis='y', style='plain')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig("co2emmission_linplot.png")
    plt.show()


def plot_agriculture_trends(df_new_year):
    """
    bar chart - show the agricultural land area for a subset of countries for 
    the years 2010 to 2015

    Parameters:
    df_new_year : DataFrame
    """
    barchart = df_new_year[df_new_year["Indicator Name"] ==
                           "Agricultural land (sq. km)"].reset_index(drop=True)
    barchart = barchart.iloc[1:10, :]
    plt.figure()

    # width
    w = 0.1
    bar1 = barchart["2010"]
    bar2 = barchart["2011"]
    bar3 = barchart["2012"]
    bar4 = barchart["2013"]
    bar5 = barchart["2014"]
    bar6 = barchart["2015"]

    x = barchart["Country Name"]
    x = np.arange(len(x))

    plt.figure(figsize=(9, 6), facecolor="lightgray")

    # Plot the barchart
    plt.bar(x, bar1, width=w, color='#8bd3c7', label="2010")
    x = x+w
    plt.bar(x, bar2, width=w, color='#fdcce5', label="2011")
    x = x+w
    plt.bar(x, bar3, width=w, color='#beb9db', label="2012")
    x = x+w
    plt.bar(x, bar4, width=w, color='#ffee65', label="2013")
    x = x+w
    plt.bar(x, bar5, width=w, color='#ffb55a', label="2014")
    x = x+w
    plt.bar(x, bar6, width=w, color='#bd7ebe', label="2015")
    x = x+w

    # set the current tick labels of the x-axis
    plt.xticks(x, ['Australia', 'Austria', 'Azerbaijan', 'Burundi',
               'Belgium', 'Benin', 'Burkina Faso', 'Bangladesh', 'Bulgaria'], size=9)

    # Labeling
    plt.xlabel("Country")
    plt.ylabel("Agriculture land area(sq.km)")

    # Chart Title
    plt.title("Agricultural Land Area by Country", size=12, fontweight='bold')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig('Agriculture_barchart.png')
    plt.show()


def plot_co2emmision(df_new_year):
    """
    bar chart - show the CO2 emission for a subset of countries for the years 
    2010 to 2015
    Parameters:
    df_new_year : DataFrame
    """
    barchart = df_new_year[df_new_year["Indicator Name"]
                           == "CO2 emissions (kt)"].reset_index(drop=True)
    barchart = barchart.iloc[1:10, :]
    plt.figure()

    # width
    w = 0.1
    bar1 = barchart["2010"]
    bar2 = barchart["2011"]
    bar3 = barchart["2012"]
    bar4 = barchart["2013"]
    bar5 = barchart["2014"]
    bar6 = barchart["2015"]

    x = barchart["Country Name"]
    x = np.arange(len(x))
    plt.figure(figsize=(9, 6), facecolor="lightgray")

    # Plot the bar chart
    plt.bar(x, bar1, width=w, color='#e60049', label="2010")
    x = x+w
    plt.bar(x, bar2, width=w, color='#0bb4ff', label="2011")
    x = x+w
    plt.bar(x, bar3, width=w, color='#50e991', label="2012")
    x = x+w
    plt.bar(x, bar4, width=w, color='#e6d800', label="2013")
    x = x+w
    plt.bar(x, bar5, width=w, color='#9b19f5', label="2014")
    x = x+w
    plt.bar(x, bar6, width=w, color='#ffa300', label="2015")
    x = x+w

    # set the current tick labels of the x-axis
    plt.xticks(x, ['Australia', 'Austria', 'Azerbaijan', 'Burundi',
               'Belgium', 'Benin', 'Burkina Faso', 'Bangladesh', 'Bulgaria'], size=9)

    # Labeling
    plt.xlabel("Country")
    plt.ylabel("CO2 emissions (kt)")

    # Chart Title
    plt.title("Co2 emmision by Country", size=12, fontweight='bold')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig("co2emission_barchart.png")
    plt.show()


def plot_heatmap(df_new_year_Australia, df_new_year_Bangladesh,
                 df_new_year_Burundidf_new_year_Burundi):
    """
    Plots a correlation heatmap for each given dataframe

    Parameters:
    df_new_year_Australia: DataFrame
        A DataFrame containing the data for Australia for different years.

    df_new_year_Bangladesh: DataFrame
        A DataFrame containing the data for Bangladesh for different years.

    df_new_year_Burundi: DataFrame
        A DataFrame containing the data for Burundi for different years.
    """
    # Australia heatmap
    plt.figure(figsize=(9, 6), facecolor="white")
    A_heatmap = plt.imshow(df_new_year_Australia.corr(), cmap="RdYlGn")
    plt.xticks(np.arange(len(df_new_year_Australia.columns)),
               df_new_year_Australia.columns, rotation=45)
    plt.yticks(np.arange(len(df_new_year_Australia.columns)),
               df_new_year_Australia.columns)
    plt.xlabel("Indicators")
    plt.ylabel("Indicators")

    # add colorbar legend to the heatmap
    plt.colorbar(A_heatmap)

    # add a title to the plot
    plt.title("Correlation between indicators - Australia",
              fontweight='bold', fontsize='10')
    plt.savefig('Australia_heatmap.png')
    plt.show()

    # Bangladesh heatmap
    plt.figure(figsize=(9, 6), facecolor="white")
    B_heatmap = plt.imshow(df_new_year_Bangladesh.corr(), cmap="coolwarm")
    plt.xticks(np.arange(len(df_new_year_Bangladesh.columns)),
               df_new_year_Bangladesh.columns, rotation=45)
    plt.yticks(np.arange(len(df_new_year_Bangladesh.columns)),
               df_new_year_Bangladesh.columns)
    plt.xlabel("Indicators")
    plt.ylabel("Indicators")

    # add colorbar legend to the heatmap
    plt.colorbar(B_heatmap)

    # add a title to the plot
    plt.title("Correlation between indicators - Bangladesh")
    plt.savefig('Bangladesh_heatmap.png')
    plt.show()

    # Burundi heatmap
    plt.figure(figsize=(9, 6), facecolor="white")
    Bur_heatmap = plt.imshow(df_new_year_Burundi.corr(), cmap="plasma")
    plt.xticks(np.arange(len(df_new_year_Burundi.columns)),
               df_new_year_Burundi.columns, rotation=45)
    plt.yticks(np.arange(len(df_new_year_Burundi.columns)),
               df_new_year_Burundi.columns)
    plt.xlabel("Indicators")
    plt.ylabel("Indicators")

    # add colorbar legend to the heatmap
    plt.colorbar(Bur_heatmap)

    # add a title to the plot
    plt.title("Correlation between indicators - Burundi")
    plt.savefig('Burundi_heatmap.png')
    plt.show()


def co2_emission_percapita(df_new_year_Bangladesh, df_new_year_Australia,
                           df_new_year_Bahrain, df_new_year_Burundi):
    """
    Calculates the CO2 emissions per capita for the given countries and creates 
    a table showing the values for each year for all the countries.

    Parameters:
    df_new_year_Bangladesh: DataFrame containing the indicators data for Bangladesh.
    df_new_year_Australia: DataFrame containing the indicators data for Australia.
    df_new_year_Bahrain: DataFrame containing the indicators data for Bahrain.
    df_new_year_Burundi: DataFrame containing the indicators data for Burundi.

    Returns:
    df_concat: A pandas DataFrame containing the CO2 emissions per capita values

    """
    # Divide CO2 emmision of each country by population in thousands and round up to two decimal points
    df_new_year_Bangladesh['CO2 emissions per capita'] = (
        df_new_year_Bangladesh['CO2 emissions (kt)'] /
        (df_new_year_Bangladesh['Urban population'] / 1000)).round(2)
    df_new_year_Australia['CO2 emissions per capita'] = (
        df_new_year_Australia['CO2 emissions (kt)'] /
        (df_new_year_Australia['Urban population'] / 1000)).round(2)
    df_new_year_Bahrain['CO2 emissions per capita'] = (
        df_new_year_Bahrain['CO2 emissions (kt)'] /
        (df_new_year_Bahrain['Urban population'] / 1000)).round(2)
    df_new_year_Burundi['CO2 emissions per capita'] = (
        df_new_year_Burundi['CO2 emissions (kt)'] /
        (df_new_year_Burundi['Urban population'] / 1000)).round(2)

    # Select only "CO2 emissions per capita" column from each dataframe
    df_Ban = df_new_year_Bangladesh.iloc[25:, 3:]

    # Rename the column by country name
    df_Ban = df_Ban.rename(columns={"CO2 emissions per capita": "Bangladesh"})
    df_Aus = df_new_year_Australia.iloc[25:, 3:]

    # Rename the column by country name
    df_Aus = df_Aus.rename(columns={"CO2 emissions per capita": "Australia"})
    df_Bah = df_new_year_Bahrain.iloc[25:, 3:]

    # Rename the column by country name
    df_Bah = df_Bah.rename(columns={"CO2 emissions per capita": "Bahrain"})
    df_Bur = df_new_year_Burundi.iloc[25:, 3:]

    # Rename the column by country name
    df_Bur = df_Bur.rename(columns={"CO2 emissions per capita": "Burundi"})

    # Concatanate dataframes into one Dataframe
    df_concat = pd.concat([df_Ban, df_Aus, df_Bah, df_Bur], axis=1)
    df_concat = df_concat.reset_index(drop=False)

    # Rename index column as "Year"
    df_concat = df_concat.rename(columns={'index': 'Year'})
    plt.subplots(figsize=(10, 9))
    plt.table(cellText=df_concat.values,
              colLabels=df_concat.columns, loc='center')
    plt.title("Co2 Emission per capita")
    plt.axis('off')
    plt.savefig('table.png')
    return df_concat


def print_details(df_new_year_Bangladesh, df_new_year_Australia,
                  df_new_year_Bahrain, df_new_year_Burundi, corr_bangladesh,
                  corr_australia, corr_bahrain, corr_burundi, df_concat):
    """
    Prints the summary statistics, correlation analysis and CO2 emissions per 
    capita for the given dataframes and correlations.

    Parameters:

    df_new_year_Bangladesh : A dataframe containing the data for Bangladesh
    df_new_year_Australia : A dataframe containing the data for Australia
    df_new_year_Bahrain : A dataframe containing the data for Bahrain
    df_new_year_Burundi :A dataframe containing the data for Burundi 
    corr_bangladesh : A correlation matrix for Bangladesh
    corr_australia : A correlation matrix for Australia
    corr_bahrain : A correlation matrix for Bahrain
    corr_burundi : A correlation matrix for Burundi
    df_concat : A dataframe containing the CO2 emissions per capita for all countries
    """
    # Describe - Summary Statistics
    print("Bangladesh Summary Statistics\n", df_new_year_Bangladesh.describe())
    print("Australia Summary Statistics\n", df_new_year_Australia.describe())
    print("Bahrain Summary Statistics\n", df_new_year_Bahrain.describe())
    print("Burundi Summary Statistics\n", df_new_year_Burundi.describe())

    # print statistics data in a dictionary
    df_statistics

    # print correlation analysis
    print("Correlation between indicators - Bangladesh:", corr_bangladesh)
    print("Correlation between indicators - Australia:", corr_australia)
    print("Correlation between indicators - Bahrain:", corr_bahrain)
    print("Correlation between indicators - Burundi:", corr_burundi)

    # print co2 emmisions per capita
    print("co2 emmisions per capita", df_concat)


# Function caller
df_new_country, df_new_year = read_file("Climate_Change.csv")
df_new_year_Bangladesh, df_new_year_Australia, df_new_year_Bahrain, df_new_year_Burundi = retrieve_countrydata(
    df_new_country)
df_statistics = calculate_statistics(
    df_new_year_Bangladesh, df_new_year_Australia, df_new_year_Bahrain, df_new_year_Burundi)
corr_bangladesh, corr_australia, corr_bahrain, corr_burundi = correlation_calculation(
    df_new_year_Bangladesh, df_new_year_Australia, df_new_year_Bahrain, df_new_year_Burundi)
df_new_country_multiindex = create_multiindex_columns(df_new_country)
plot_population_trends(df_new_country_multiindex)
plot_co2emmision_overtime(df_new_country_multiindex)
plot_agriculture_trends(df_new_year)
plot_co2emmision(df_new_year)
plot_heatmap(df_new_year_Australia,
             df_new_year_Bangladesh, df_new_year_Burundi)
df_concat = co2_emission_percapita(
    df_new_year_Bangladesh, df_new_year_Australia, df_new_year_Bahrain, df_new_year_Burundi)
print_details(df_new_year_Bangladesh, df_new_year_Australia, df_new_year_Bahrain,
              df_new_year_Burundi, corr_bangladesh, corr_australia, corr_bahrain, corr_burundi, df_concat)
