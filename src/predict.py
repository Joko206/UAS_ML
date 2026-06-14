import os
import pandas as pd
import joblib

def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'best_model.pkl')
    
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"Error: Model tidak ditemukan di {model_path}. Harap jalankan train.py terlebih dahulu.")
        return None

def predict_single_data(data_dict):
    """
    Melakukan prediksi untuk satu data baru
    data_dict harus berupa dictionary dengan key yang sesuai dengan nama kolom.
    """
    model = load_model()
    if model is None:
        return
    
    # Konversi input ke DataFrame
    df_input = pd.DataFrame([data_dict])
    
    # Prediksi
    prediction = model.predict(df_input)
    probability = model.predict_proba(df_input)
    
    print("\n--- Hasil Prediksi ---")
    print("Input Data:")
    print(df_input.to_string(index=False))
    
    print(f"\nPrediksi Kelas: {'Penyakit Jantung' if prediction[0] == 1 else 'Normal (Bukan Penyakit Jantung)'}")
    print(f"Probabilitas Penyakit Jantung: {probability[0][1] * 100:.2f}%")

if __name__ == "__main__":
    # Contoh data input baru untuk diprediksi
    sample_data = {
        'Age': 50,
        'Sex': 'M',
        'ChestPainType': 'ATA',
        'RestingBP': 140,
        'Cholesterol': 289,
        'FastingBS': 0,
        'RestingECG': 'Normal',
        'MaxHR': 172,
        'ExerciseAngina': 'N',
        'Oldpeak': 0.0,
        'ST_Slope': 'Up'
    }
    
    print("Mencoba prediksi menggunakan data dummy...")
    predict_single_data(sample_data)
