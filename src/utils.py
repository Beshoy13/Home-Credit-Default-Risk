import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def set_abs_columns(data, date_related_columns):
    for col in date_related_columns:
        
        data[col] = abs(data[col])

        # Descriptive statistics
        print(f"Done for: {col}")
    
def MiniMaxScaler_data(data, columns_to_normalize):
    # Apply Min-Max Normalization
    scaler = MinMaxScaler()
    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
    
def Split_Train_Test(data, target_column):
    """
    This is for Split data into x,y and split it again into train and test
    """
    x = data.drop(columns=[target_column])
    y = data[target_column]
    
    return train_test_split(x, y, stratify=y, train_size=0.8)
