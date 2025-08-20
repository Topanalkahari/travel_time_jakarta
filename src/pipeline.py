from src.preprocessor import TravelTimePreprocessor

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


class TravelTimePipeline:
    """
    Complete pipeline untuk travel time prediction
    """
    
    def __init__(self, model=None, test_size=0.2, random_state=23):
        self.preprocessor = TravelTimePreprocessor()
        self.model = model if model is not None else LinearRegression()
        self.test_size = test_size
        self.random_state = random_state
        self.is_fitted = False
        
    def load_data(self, filepath):
        """Load data dari CSV file"""
        self.data = pd.read_csv(filepath)
        print(f"✅ Data berhasil dimuat: {self.data.shape}")
        return self.data
    
    def prepare_data(self, target_column='travel_time'):
        """Prepare data untuk training"""
        if self.data is None:
            raise ValueError("Data belum dimuat. Gunakan load_data() terlebih dahulu.")
        
        # Separate features and target
        y = self.data[target_column]
        X = self.data.drop([target_column], axis=1)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"✅ Data berhasil dipisah:")
        print(f"   Training: {self.X_train.shape}")
        print(f"   Testing: {self.X_test.shape}")
        print(f"📊 Jumlah fitur sebelum encoding: {self.X_train.shape[1]}")
        print("Daftar fitur sebelum encoding:", list(self.X_train.columns))
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def fit(self, X=None, y=None):
        """Fit pipeline pada training data"""
        if X is None or y is None:
            X, y = self.X_train, self.y_train
        
        # Fit preprocessor
        print("🔄 Melakukan preprocessing...")
        self.preprocessor.fit(X, y)
        
        # Transform training data
        X_processed = self.preprocessor.transform(X, y)
        
        # Fit model
        print("🔄 Melatih model...")
        self.model.fit(X_processed, y)
        
        self.is_fitted = True
        print("✅ Pipeline berhasil dilatih!")
        
        return self
    
    def predict(self, X):
        """Prediksi menggunakan fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline belum dilatih. Gunakan fit() terlebih dahulu.")
        
        # Preprocess dan prediksi
        X_processed = self.preprocessor.transform(X)
        predictions = self.model.predict(X_processed)
        
        return predictions
    
    def evaluate(self, X=None, y=None, dataset_name="Test"):
        """Evaluasi model pada dataset"""
        if X is None or y is None:
            X, y = self.X_test, self.y_test
        
        # Prediksi
        y_pred = self.predict(X)
        
        # Hitung metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Tampilkan hasil
        print(f"📊 Evaluasi Model pada Data {dataset_name}:")
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"R²   : {r2:.4f}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mse': mse,
            'r2': r2,
            'predictions': y_pred
        }
    
    def save_pipeline(self, filepath):
        """Simpan pipeline yang sudah dilatih"""
        if not self.is_fitted:
            raise ValueError("Pipeline belum dilatih. Gunakan fit() terlebih dahulu.")
        
        pipeline_data = {
            'preprocessor': self.preprocessor,
            'model': self.model,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(pipeline_data, filepath)
        print(f"✅ Pipeline berhasil disimpan ke: {filepath}")
    
    def load_pipeline(self, filepath):
        """Load pipeline yang sudah dilatih"""
        pipeline_data = joblib.load(filepath)
        
        self.preprocessor = pipeline_data['preprocessor']
        self.model = pipeline_data['model']
        self.is_fitted = pipeline_data['is_fitted']
        
        print(f"✅ Pipeline berhasil dimuat dari: {filepath}")
    
    def get_feature_importance(self, top_n=10):
        """Dapatkan feature importance (untuk LinearRegression menggunakan koefisien)"""
        if not self.is_fitted:
            raise ValueError("Pipeline belum dilatih. Gunakan fit() terlebih dahulu.")
        
        if hasattr(self.model, 'coef_'):
            # Get processed feature names
            X_sample = self.preprocessor.transform(self.X_train.iloc[:1])
            feature_names = X_sample.columns
            
            # Get feature importance (absolute coefficients)
            importance = np.abs(self.model.coef_)
            
            # Create DataFrame
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print(f"📈 Top {top_n} Feature Importance:")
            print(feature_importance.head(top_n))
            print(f"Total fitur setelah encoding: {len(feature_names)}")
            
            
            return feature_importance
        else:
            print("❌ Model tidak memiliki feature importance")
            return None