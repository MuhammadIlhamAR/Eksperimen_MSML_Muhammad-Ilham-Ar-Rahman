import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def run_automation():
    # Gunakan path relatif dari root repository agar GitHub Actions tidak bingung
    input_path = "PRSA_Data_Aotizhongxin_raw/PRSA_Data_Aotizhongxin.csv"
    output_dir = "preprocessing/namadataset_preprocessing"
    output_file = f"{output_dir}/PRSA_Data_Aotizhongxin_preprocessed.csv"

    # Proses Preprocessing (Sesuai notebook eksperimen lo)
    df = pd.read_csv(input_path)
    df = df.drop(columns=['No', 'station'], errors='ignore')
    
    # Isi missing values & encoding (Skilled level)
    df['RAIN_Category'] = np.where(df['RAIN'] == 0, 'Tidak Hujan', 'Hujan')
    # ... (tambahkan logika pembersihan lainnya sesuai notebook lo)
    
    # Simpan hasil
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_csv(output_file, index=False)
    print(f"✅ Berhasil! Dataset tersimpan di {output_file}")

if __name__ == "__main__":
    run_automation()
