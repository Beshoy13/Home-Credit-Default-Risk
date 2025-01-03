from src.model import XGBoost_Model, save_model_h5, load_model_h5, Train_model, train_and_evaluate_xgboost
from src.preprocess import Read_csv_Data, Drop_Null_Columns, Drop_Null_Rows, Remove_Duplicated_Rows, Fill_Home_Risk_Null_Values, Apply_Label_Encoding, Apply_OneHot_Encoding
from src.utils import MiniMaxScaler_data, Split_Train_Test, set_abs_columns

raw_data = 'data/raw/application_train.csv'
model_path = 'model_1.h5'

if __name__ == "__main__":
    print("Start Train")
    
    # read training data
    data = Read_csv_Data(raw_data)
    
    # Drop columns
    Drop_Null_Columns(data, 58.0)
    
    # Drop Rows in class 0 and have more than 30 null columns
    data = Drop_Null_Rows(data, 30, 'TARGET', 0)
    
    # Remove duplicated values
    Remove_Duplicated_Rows(data)
    
    # Fill null values
    Fill_Home_Risk_Null_Values(data)
    
    # Remove negative values
    date_related_columns = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION' ,"DAYS_ID_PUBLISH",]
    set_abs_columns(data, date_related_columns)
    
    
    # Normalizations for the following columns
    # Columns to normalize
    columns_to_normalize = [
        'SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
        'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
        'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH'
    ]
    
    MiniMaxScaler_data(data, columns_to_normalize)
    
    
    # Columns to encode
    label_encoding_columns = ['NAME_EDUCATION_TYPE', 'WEEKDAY_APPR_PROCESS_START']
    one_hot_encoding_columns = [
        'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE',
        'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
    ]
    
    Apply_Label_Encoding(data, label_encoding_columns)
    data = Apply_OneHot_Encoding(data, one_hot_encoding_columns)
    print("Done Apply Encoding..")
    
    
    # split data
    x_train, x_test, y_train, y_test = Split_Train_Test(data, 'TARGET')
    
    # Create xgboost model
    model_xgb = XGBoost_Model(y_train)
    
    # Start training model
    print("Start trainning model")
    train_and_evaluate_xgboost(model_xgb, x_train, y_train, x_test, y_test)
    
    # save model
    print(f"Saving model to {model_path}")
    save_model_h5(model_xgb, model_path)