import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

def Read_csv_Data(path):
    """
    Function to read Data in csv format using pandas
    """
    return pd.read_csv(path)
    
    
    
def Drop_Null_Columns(data, threshold = 60.0):
    """
    Function to Drop columns that have null percentage bigger than threshold (60.0)
    """
    columns_list = data.columns.tolist()


    for col in columns_list:
      null_percentage = (data[col].isnull().sum() / len(data)) * 100
      if null_percentage >= threshold:
        print(f"{col}: {null_percentage} --- Droped")
        data.drop(columns=[col], inplace=True)
        

def Drop_Null_Rows(data, null_columns = 10, target = 'TARGET', value = 0):
    null_counts = data.isnull().sum(axis=1)
    rows_to_remove = (null_counts > 30) & (data['TARGET'] == 0)
    return data[~rows_to_remove]
    print(f"Number of Rows Removed: {rows_to_remove.sum()}")

def Remove_Duplicated_Rows(data):
    """
    This function is for drop duplicated Rows
    """
    # Check the number of rows before and after
    original_rows = len(data)
    data = data.drop_duplicates()
    removed_rows = original_rows - len(data)
    print(f"Number of duplicate rows removed: {removed_rows}")
    


def Fill_Null_Values(data):
    # numerical coloumns
    missforest_imputer=IterativeImputer(random_state=40)
    numerical_coloumns= data.select_dtypes(include=["Float64","int64"]).columns
    data[numerical_coloumns]= missforest_imputer.fit_transform(data[numerical_coloumns])

    #cattegorical coloumns

    SimpleImputer=SimpleImputer(strategy="most_frequent")
    cateogrical_coloumns=data.select_dtypes(include=["object"]).columns
    data[cateogrical_coloumns]=SimpleImputer.fit_transform(data[cateogrical_coloumns])

    print(f"number of nans in numerical coloumns: {data[numerical_coloumns].isnull().sum()}")
    print(f"number of nans in cateogrical coloumns: {data[cateogrical_coloumns].isnull().sum()}")
    
    
def Apply_OneHot_Encoding(data, one_hot_encoding_columns):
    # Apply one-hot encoding
    return pd.get_dummies(data, columns=one_hot_encoding_columns, drop_first=False)
    
def Apply_Label_Encoding(data, label_encoding_columns):
    # Initialize a label encoder
    label_encoder = LabelEncoder()
    
    # Apply label encoding
    for col in label_encoding_columns:
        data[col] = label_encoder.fit_transform(data[col])

    
def Fill_Home_Risk_Null_Values(data):
    # Iterative Imputer (MICE) for EXT_SOURCE_1
    iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
    data[['EXT_SOURCE_1']] = iterative_imputer.fit_transform(data[['EXT_SOURCE_1']])
    
    
    # Median/Mean Imputation for NONLIVINGAREA_*, ELEVATORS_*, APARTMENTS_*, ENTRANCES_*, LIVINGAREA_*
    median_imputation_columns = [
        'NONLIVINGAREA_AVG', 'ELEVATORS_AVG', 'APARTMENTS_AVG',
        'ENTRANCES_AVG', 'LIVINGAREA_AVG'
    ]
    for col in median_imputation_columns:
        data[col] = data[col].fillna(data[col].median())  # Replace with .mean() if preferred
        
        
    # Fill with 'Missing' for WALLSMATERIAL_MODE, HOUSETYPE_MODE, EMERGENCYSTATE_MODE
    categorical_columns = ['WALLSMATERIAL_MODE', 'HOUSETYPE_MODE', 'EMERGENCYSTATE_MODE']
    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])
        
    # Iterative Imputer (MICE) for YEARS_BEGINEXPLUATATION_*
    data[['YEARS_BEGINEXPLUATATION_AVG']] = iterative_imputer.fit_transform(data[['YEARS_BEGINEXPLUATATION_AVG']])
    
    
    # Iterative Imputer for TOTALAREA_MODE
    data[['TOTALAREA_MODE']] = iterative_imputer.fit_transform(data[['TOTALAREA_MODE']])
    
    # Mode Imputation for OCCUPATION_TYPE (Faster Alternative)
    data['OCCUPATION_TYPE'] = data['OCCUPATION_TYPE'].fillna(data['OCCUPATION_TYPE'].mode()[0])
    # Mode Imputation with Conditional Probability for NAME_TYPE_SUITE
    data['NAME_TYPE_SUITE'] = data['NAME_TYPE_SUITE'].fillna(data['NAME_TYPE_SUITE'].mode()[0])
    
    # Median Imputation for AMT_REQ_CREDIT_BUREAU_*
    credit_columns = [
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]
    for col in credit_columns:
        data[col] = data[col].fillna(data[col].median())

    # Median Imputation for DEF_30_CNT_SOCIAL_CIRCLE, OBS_30_CNT_SOCIAL_CIRCLE
    social_columns = [
        'DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
        'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE'
    ]
    for col in social_columns:
        data[col] = data[col].fillna(data[col].median())
        
        
    high_missing_columns = [
        'NONLIVINGAREA_MODE', 'NONLIVINGAREA_MEDI', 'ELEVATORS_MODE', 'ELEVATORS_MEDI',
        'APARTMENTS_MODE', 'APARTMENTS_MEDI', 'ENTRANCES_MEDI', 'ENTRANCES_MODE',
        'LIVINGAREA_MODE', 'LIVINGAREA_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMAX_AVG', 'FLOORSMAX_MODE'
    ]
    for col in high_missing_columns:
        data[col] = data[col].fillna(data[col].median())
        
    iterative_columns = [
        'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BEGINEXPLUATATION_MEDI', 'EXT_SOURCE_3'
    ]
    data[iterative_columns] = iterative_imputer.fit_transform(data[iterative_columns])


    # Iterative Imputer for EXT_SOURCE_2, AMT_GOODS_PRICE, CNT_FAM_MEMBERS
    low_missing_columns = ['EXT_SOURCE_2', 'AMT_GOODS_PRICE', 'CNT_FAM_MEMBERS']
    data[low_missing_columns] = iterative_imputer.fit_transform(data[low_missing_columns])


    # Fill DAYS_LAST_PHONE_CHANGE with Median
    data['DAYS_LAST_PHONE_CHANGE'] = data['DAYS_LAST_PHONE_CHANGE'].fillna(data['DAYS_LAST_PHONE_CHANGE'].median())


    data['AMT_ANNUITY'] = data['AMT_ANNUITY'].fillna(data['AMT_ANNUITY'].median())
    
    print("############## Done Fill Null Values ###################")
    
    # Check no null values
    percent_nan = data.isnull().sum()/len(data)*100
    print(f"Sum of Null: {percent_nan.sum()}")