import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib

def plot_learning_curve(estimator, title, X, y, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), save_path=None):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score (Accuracy)")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    print("Memulai proses eksperimen pemodelan...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'heart.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Data
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset dimuat: {len(df)} baris.")
    except FileNotFoundError:
        print(f"Error: Dataset tidak ditemukan di {data_path}")
        return

    # Menangani nilai '0' yang tidak masuk akal pada fitur klinis dengan mengubahnya menjadi NaN (Missing Values)
    # Fitur ini akan ditangani secara otomatis oleh imputer di dalam Pipeline untuk mencegah Data Leakage
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df['RestingBP'] = df['RestingBP'].replace(0, np.nan)
    
    # Pisahkan fitur dan target
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']
    
    numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    
    # 2. Preprocessing
    # Penggunaan Pipeline secara ketat mencegah terjadinya 'Data Leakage'
    # karena Imputasi (Median/Mode) dihitung secara eksklusif menggunakan set Training.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 3. Data Splitting (Train, Validation, Test)
    # 70% Train, 15% Validation, 15% Test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1764, random_state=42) # 0.1764 dari 85% ~ 15% dari total
    
    print(f"Ukuran Data - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # --- MODEL 1: BASELINE (Logistic Regression) ---
    print("\n[1] Melatih Baseline Model: Logistic Regression")
    baseline_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    baseline_pipeline.fit(X_train, y_train)
    y_val_pred_base = baseline_pipeline.predict(X_val)
    print(f"Baseline Validation Accuracy: {accuracy_score(y_val, y_val_pred_base):.4f}")
    
    # --- MODEL 2: KOMPLEKS (Random Forest) dg Hyperparameter Tuning ---
    print("\n[2] Melatih Model Kompleks: Random Forest dengan GridSearchCV")
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Parameter Terbaik Random Forest: {grid_search.best_params_}")
    
    y_val_pred_rf = best_model.predict(X_val)
    print(f"Optimized Random Forest Validation Accuracy: {accuracy_score(y_val, y_val_pred_rf):.4f}")
    
    # --- EVALUASI PADA TEST SET MENGGUNAKAN MODEL TERBAIK ---
    print("\n[3] Evaluasi Model Terbaik pada Data Uji (Test Set)")
    y_test_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nLaporan Lengkap:")
    print(classification_report(y_test, y_test_pred))
    
    # --- LEARNING CURVE ---
    print("\n[4] Membuat Learning Curve...")
    curve_path = os.path.join(models_dir, 'learning_curve.png')
    plot_learning_curve(best_model, "Learning Curve (Random Forest)", X_train, y_train, cv=5, save_path=curve_path)
    print(f"Grafik Learning Curve disimpan di: {curve_path}")
    
    # Simpan Model
    model_save_path = os.path.join(models_dir, 'best_model.pkl')
    joblib.dump(best_model, model_save_path)
    print(f"\nModel terbaik berhasil disimpan di: {model_save_path}")

if __name__ == "__main__":
    main()
