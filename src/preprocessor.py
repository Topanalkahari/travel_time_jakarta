import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class TravelTimePreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom preprocessor untuk data travel time prediction
    """
    
    def __init__(self):
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.target_encoders = {}
        self.global_mean = None
        
    def _clean_location_names(self, df):
        """Membersihkan nama lokasi dari informasi tambahan dalam kurung"""
        df_clean = df.copy()
        df_clean['start_point'] = df_clean['start_point'].str.split(' (', regex=False).str[0]
        df_clean['end_point'] = df_clean['end_point'].str.split(' (', regex=False).str[0]
        return df_clean
    
    def _create_route_features(self, df):
        """Membuat fitur route dan kombinasi lainnya"""
        df_new = df.copy()
        
        # Buat route
        df_new['route'] = df_new['start_point'] + ' to ' + df_new['end_point']
        
        # Drop kolom asli
        df_new = df_new.drop(['start_point', 'end_point'], axis=1)
        
        # Lowercase day_of_week
        df_new['day_of_week'] = df_new['day_of_week'].str.lower()
        
        # Buat fitur is_weekend
        df_new['is_weekend'] = df_new['day_of_week'].apply(
            lambda x: 1 if x in ['saturday', 'sunday'] else 0
        )
        
        # Buat kombinasi fitur
        df_new['route_day'] = df_new['route'] + '_' + df_new['day_of_week']
        df_new['route_time'] = df_new['route'] + '_' + df_new['time_of_day']
        df_new['route_day_time'] = df_new['route'] + '_' + df_new['day_of_week'] + '_' + df_new['time_of_day']
        
        return df_new
    
    def _handle_missing_values(self, df, is_training=True):
        """Menangani missing values"""
        df_new = df.copy()
        
        # Handle categorical columns with 'nan' string
        obj_nan = ['vehicle_density', 'population_density', 'weather']
        for col in obj_nan:
            df_new[col] = df_new[col].str.lower()
            df_new[col] = df_new[col].replace('nan', np.nan)
            df_new[col] = df_new[col].fillna('missing')
        
        # Handle traffic_condition
        df_new['traffic_condition_missing'] = df_new['traffic_condition'].isna().astype(int)
        
        if is_training:
            self.traffic_condition_mode = df_new['traffic_condition'].mode()[0]
        
        df_new['traffic_condition'] = df_new['traffic_condition'].fillna(self.traffic_condition_mode)
        
        return df_new
    
    def _transform_features(self, df):
        """Transformasi fitur numerik"""
        df_new = df.copy()
        
        # Log transform event_count
        df_new['event_count'] = np.log1p(df_new['event_count'])
        
        return df_new
    
    def fit(self, X, y=None):
        """Fit preprocessor pada training data"""
        # Store global mean for target encoding
        if y is not None:
            self.global_mean = y.mean()
        
        # Apply preprocessing steps
        X_processed = self._clean_location_names(X)
        X_processed = self._create_route_features(X_processed)
        X_processed = self._handle_missing_values(X_processed, is_training=True)
        X_processed = self._transform_features(X_processed)
        
        # Separate numerical and categorical features
        X_num = X_processed._get_numeric_data()
        X_cat = X_processed.drop(list(X_num.columns.values), axis=1)
        
        # Fit OneHotEncoder pada categorical features tertentu
        cat_features_for_ohe = ['time_of_day', 'day_of_week', 'vehicle_density', 
                               'population_density', 'weather', 'route']
        self.cat_features_for_ohe = cat_features_for_ohe
        
        if all(col in X_cat.columns for col in cat_features_for_ohe):
            self.ohe.fit(X_cat[cat_features_for_ohe])
        
        # Fit target encoder untuk high-cardinality features
        target_encode_cols = ['route_day', 'route_time', 'route_day_time']
        self.target_encode_cols = target_encode_cols
        
        if y is not None:
            temp_df = X_processed.copy()
            temp_df['target'] = y.values
            
            for col in target_encode_cols:
                if col in temp_df.columns:
                    self.target_encoders[col] = temp_df.groupby(col)['target'].mean()
        
        # Prepare untuk scaling - simpan kolom numerik setelah encoding
        X_encoded = self._apply_encoding(X_processed, y, is_training=True)
        self.num_features = X_encoded.select_dtypes(include=['float64', 'int64']).columns
        
        # Fit scaler
        self.scaler.fit(X_encoded[self.num_features])
        
        return self
    
    def _apply_encoding(self, X_processed, y=None, is_training=False):
        """Apply encoding to features"""
        # Separate numerical and categorical
        X_num = X_processed._get_numeric_data()
        X_cat = X_processed.drop(list(X_num.columns.values), axis=1)
        
        # OneHotEncode specific categorical features
        if all(col in X_cat.columns for col in self.cat_features_for_ohe):
            cat_encoded = self.ohe.transform(X_cat[self.cat_features_for_ohe])
            cat_encoded_df = pd.DataFrame(
                cat_encoded, 
                columns=self.ohe.get_feature_names_out(self.cat_features_for_ohe),
                index=X_cat.index
            )
            
            # Remove original categorical columns and add encoded ones
            X_cat = X_cat.drop(self.cat_features_for_ohe, axis=1)
            X_cat = pd.concat([X_cat, cat_encoded_df], axis=1)
        
        # Target encoding untuk high-cardinality features
        X_encoded = X_cat.copy()
        
        for col in self.target_encode_cols:
            if col in X_encoded.columns:
                X_encoded[col] = X_cat[col].map(self.target_encoders[col]).fillna(self.global_mean)
        
        # Combine numerical and categorical
        X_final = pd.concat([X_num, X_encoded], axis=1)
        
        return X_final
    
    def transform(self, X, y=None):
        """Transform data menggunakan fitted preprocessor"""
        # Apply preprocessing steps
        X_processed = self._clean_location_names(X)
        X_processed = self._create_route_features(X_processed)
        X_processed = self._handle_missing_values(X_processed, is_training=False)
        X_processed = self._transform_features(X_processed)
        
        # Apply encoding
        X_encoded = self._apply_encoding(X_processed, y, is_training=False)
        
        # Scale numerical features
        X_scaled = X_encoded.copy()
        X_scaled[self.num_features] = self.scaler.transform(X_encoded[self.num_features])
        
        return X_scaled