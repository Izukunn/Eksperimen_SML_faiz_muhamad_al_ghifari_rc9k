import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'File tidak ditemukan di: {path}')
    df = pd.read_csv(path)
    print(f'Data dimuat. Shape awal: {df.shape}')
    return df

def preprocess_data(df):
    target_col = 'condition'

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['cp', 'thal']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X)

    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    new_columns = numeric_features + list(ohe_feature_names) + [col for col in X.columns if col not in numeric_features + categorical_features]

    X_df = pd.DataFrame(X_processed, columns=new_columns)

    return X_df, y

def save_split_data(X, y, output_folder='data_processed'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_train.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    os.makedirs(output_folder, exist_ok=True)

    train_path = os.path.join(output_folder, 'train.csv')
    test_path = os.path.join(output_folder, 'test.csv')

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Data tersimpan di folder '{output_folder}':")
    print(f" - Train: {train_data.shape}")
    print(f" - Test: {test_data.shape}")

if __name__ == "__main__":
    input_path = '../heart_cleveland_upload.csv'

    try:
        df = load_data(input_path)
        X_clean, y = preprocess_data(df)
        save_split_data(X_clean, y, output_folder='data_processed')

        print("\n=== Otomatisasi Preprocessing Selesai ===")
    except Exception as e:
        print(f"Terjadi error: {e}")