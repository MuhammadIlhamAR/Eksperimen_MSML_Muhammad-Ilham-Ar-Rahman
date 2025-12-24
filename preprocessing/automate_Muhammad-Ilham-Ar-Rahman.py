import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def run_preprocessing(input_path, output_path):
    print(f"🚀 Memulai preprocessing data dari: {input_path}")
    
    # 1. Load Data
    df = pd.read_csv(input_path)
    
    # 2. Drop kolom yang tidak relevan
    cols_to_drop = ['No', 'station']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # 3. Handling Missing Values (Numerik: Mean, Kategorik: Mode)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    if 'wd' in df.columns:
        df['wd'] = df['wd'].fillna(df['wd'].mode()[0])

    # 4. Create RAIN_Category (Target)
    if 'RAIN' in df.columns:
        df['RAIN_Category'] = np.where(df['RAIN'] == 0, 'Tidak Hujan', 'Hujan')
        df = df.drop(columns=['RAIN'])

    # 5. Handling Outliers (IQR Method)
    cols_outlier = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']
    for col in [c for c in cols_outlier if c in df.columns]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])

    # 6. Encoding Kategorikal
    if 'wd' in df.columns:
        le = LabelEncoder()
        df['wd_encoded'] = le.fit_transform(df['wd'])
        df = df.drop(columns=['wd'])

    # 7. Feature Scaling & SMOTE (Skilled Requirement)
    X = df.drop(columns=['RAIN_Category', 'year', 'month', 'day', 'hour'], errors='ignore')
    y = df['RAIN_Category']
    
    # Map target ke numerik untuk SMOTE
    y_mapped = y.map({'Tidak Hujan': 0, 'Hujan': 1})
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y_mapped)
    
    # Gabungkan kembali
    df_final = pd.DataFrame(X_res, columns=X.columns)
    df_final['RAIN_Category'] = y_res.map({0: 'Tidak Hujan', 1: 'Hujan'})

    # 8. Simpan Dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"✅ Preprocessing Selesai! Data disimpan di: {output_path}")

if __name__ == "__main__":
    # Path disesuaikan dengan struktur Kriteria 1
    input_file = "namadataset_raw/PRSA_Data_Aotizhongxin.csv"
    output_file = "preprocessing/namadataset_preprocessing/PRSA_Data_Aotizhongxin_preprocessed.csv"
    run_preprocessing(input_file, output_file)
