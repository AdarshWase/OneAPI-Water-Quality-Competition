import joblib
import pandas as pd

encoder = joblib.load('model/ordinal_encoder.joblib')
scaler = joblib.load('model/scaler.joblib')

def fill_missing_with_mean(X):
    missing_val_columns = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc',
                           'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity',
                           'Chlorine', 'Manganese', 'Total Dissolved Solids', 'Water Temperature', 'Air Temperature']
    
    for col in missing_val_columns:
        X[col].fillna(X[col].mean(), inplace=True)
    return X

def fill_color_mapping(X):
    X['Color'].fillna('Near Colorless', inplace=True)
    color_mapping = X.groupby('Color')['Color'].transform('count') / len(X)
    X['Color'] = color_mapping
    return X

def fill_source_with_mode(X):
    X['Source'] = X['Source'].fillna('Stream')
    X['Source'] = encoder.fit_transform(X[['Source']])
    return X

def delete_non_important_columns(X):
    columns_to_keep = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Color',
                       'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity',
                       'Chlorine', 'Manganese', 'Total Dissolved Solids', 'Source',
                       'Water Temperature', 'Air Temperature']
    
    X = X.loc[:, columns_to_keep]
    return X

def create_new_columns(X):
    # Iron
    bin_edges = [0, 0.1, 1, 20]
    bin_labels = [0, 0.4, 1]
    X['Iron_Bin'] = pd.cut(X['Iron'], bins=bin_edges, labels=bin_labels)
    
    # Nitrate
    bin_edges = [0, 1, 5, 100] 
    X['Nitrate_Bin'] = pd.cut(X['Nitrate'], bins=bin_edges, labels=bin_labels)
    
    # Copper
    bin_edges = [0, 0.02, 1, 20]
    X['Copper_Bin'] = pd.cut(X['Copper'], bins=bin_edges, labels=bin_labels)
    return X

def scale_features(X):
    return scaler.transform(X)