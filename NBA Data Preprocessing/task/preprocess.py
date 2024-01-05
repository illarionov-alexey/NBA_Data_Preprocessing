import os
import requests
import pandas as pd
import re
from itertools import combinations
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def get_initial_data_path():
    # Checking ../Data directory presence
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'nba2k-full.csv' not in os.listdir('../Data'):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/nba2k-full.csv', 'wb').write(r.content)
        print('Loaded.')

    data_path = "../Data/nba2k-full.csv"
    return data_path


def clean_data(data_file_path: str) -> pd.DataFrame:
    # Load a DataFrame from the location specified in the path parameter;
    df = pd.read_csv(data_file_path)
    # Parse the b_day and draft_year features as datetime objects;
    df.b_day = pd.to_datetime(df.b_day)
    df.draft_year = pd.to_datetime(df.draft_year, format='%Y')
    # Replace the missing values in team feature with "No Team";
    df.team.fillna('No Team', inplace=True)
    # Take the height feature in meters, the height feature contains metric and customary units;
    # Take the weight feature in kg, the weight feature contains metric and customary units;
    # Remove the extraneous $ character from the salary feature;
    # Parse the height, weight, and salary features as floats
    df.height = pd.to_numeric(df.height.transform(lambda val: re.search(r'\d[.,]\d{1,2}', val).group()))
    df.weight = pd.to_numeric(df.weight.transform(lambda val: val.split('/')[1].replace('kg.', '').strip()))
    df.salary = pd.to_numeric(df.salary.transform(lambda val: val.replace('$', '').strip()), downcast='float')
    # Categorize the country feature as "USA" and "Not-USA";
    df.loc[df.country != 'USA', 'country'] = 'Not-USA'
    # Replace the cells containing "Undrafted" in the draft_round feature with the string "0"
    df.loc[df.draft_round == "Undrafted", 'draft_round'] = '0'
    return df


def feature_data(df: pd.DataFrame) -> pd.DataFrame:
    # Get the unique values in the version column of the DataFrame you got from clean_data as a year (2020, for example)
    # and parse as a datetime object;
    df.version = pd.to_datetime(df.version.transform(lambda val: val.replace('NBA2k', '')), format='%y')
    # Engineer the age feature by subtracting b_day column from version. Calculate the value as year;
    df['age'] = df.apply(lambda r: r.version.year - r.b_day.year, axis=1)
    # Engineer the experience feature by subtracting draft_year column from version. Calculate the value as year;
    df['experience'] = df.apply(lambda r: r.version.year - r.draft_year.year, axis=1)
    # Engineer the bmi (body mass index) feature from weight and height columns. The formula is bmi=w/h2
    df['bmi'] = df.weight / df.height / df.height
    # Remove the high cardinality features;
    cols_to_drop = [col for col in df.columns if str(df.dtypes[col]) == 'object' and df[col].nunique() > 50]
    # Drop the version, b_day, draft_year, weight, and height columns;
    cols_to_drop.extend(['version', 'b_day', 'draft_year', 'weight', 'height'])
    df.drop(labels=cols_to_drop, axis='columns', inplace=True)
    return df


def multicol_data(df: pd.DataFrame) -> pd.DataFrame:
    cor = df.select_dtypes('number').corr()
    cols = cor.columns.drop('salary').to_list()
    cols_to_drop = [i if cor.loc['salary', i] <= cor.loc['salary', j] else j for i, j in combinations(cols, 2) if
                    abs(cor.loc[i, j]) > 0.5]
    return df.drop(columns=cols_to_drop)


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    # Transform numerical features in the DataFrame it got from multicol_data using StandardScaler;
    scaler = StandardScaler()
    number_features = df.select_dtypes('number').drop(columns='salary')
    X = scaler.fit_transform(number_features)
    X = pd.DataFrame(X, columns=number_features.columns)
    # Transform nominal categorical variables in the DataFrame using OneHotEncoder;
    scaler = OneHotEncoder()
    Xcat = scaler.fit_transform(df.select_dtypes('object'))
    Xcat = pd.DataFrame.sparse.from_spmatrix(Xcat,columns=[col for cols in scaler.categories_ for col in cols ])
    # Concatenate the transformed numerical and categorical features in the following order: numerical features,
    # then nominal categorical features;
    #Return two objects: X, where all the features are stored, and y with the target variable.
    return X.join(Xcat), df.salary


# write your code here
if __name__ == '__main__':
    # pd.set_option('display.max_columns', 20)
    df = clean_data(get_initial_data_path())
    df = feature_data(df)
    df = multicol_data(df)
    X, y = transform_data(df)

    answer = {
        'shape': [X.shape, y.shape],
        'features': list(X.columns),
    }
    print(answer)
