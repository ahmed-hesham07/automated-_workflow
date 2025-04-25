import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Any, Tuple, List

class EquipmentAnalysis:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Dynamically prepare features based on data columns"""
        # Create a copy to avoid modifying the original dataframe
        df_processed = df.copy()
        
        # Process datetime columns
        datetime_cols = df_processed.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            df_processed[f'{col}_year'] = df_processed[col].dt.year
            df_processed[f'{col}_month'] = df_processed[col].dt.month
            df_processed[f'{col}_day'] = df_processed[col].dt.day
            df_processed = df_processed.drop(columns=[col])
        
        # Identify numeric and categorical columns
        numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        # Find cost-related columns
        cost_related = [col for col in numeric_cols if any(term in col.lower() 
                       for term in ['cost', 'price', 'maintenance'])]
        if not cost_related:
            raise ValueError("No cost or maintenance related column found")
        
        # Remove target variable from features
        target_col = cost_related[0]
        feature_cols = [col for col in df_processed.columns if col != target_col]
        
        # Prepare feature matrix
        X = df_processed[feature_cols].copy()
        
        # Encode categorical features
        for col in categorical_cols:
            if col in feature_cols:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].fillna('missing'))
        
        # Scale numeric features
        numeric_features = [col for col in X.columns if col in numeric_cols]
        if numeric_features:
            X[numeric_features] = self.scaler.fit_transform(X[numeric_features])
        
        return X, target_col

    def train_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train ML model and return performance metrics"""
        X, target_col = self.prepare_features(df)
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate feature importance
        self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        # Return metrics
        return {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'feature_importance': self.feature_importance
        }
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in equipment maintenance patterns"""
        X, _ = self.prepare_features(df)
        
        # Train isolation forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X)
        
        # Add anomaly detection results to dataframe
        df_result = df.copy()
        df_result['is_anomaly'] = anomalies == -1
        return df_result
    
    def get_business_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate business insights from the equipment maintenance data"""
        # Get only numeric cost columns
        cost_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
                    if any(term in col.lower() for term in ['cost', 'price', 'maintenance'])]
        
        equipment_cols = [col for col in df.columns if any(term in col.lower() 
                        for term in ['equipment', 'type', 'category'])]
        
        if not cost_cols or not equipment_cols:
            raise ValueError("Required cost or equipment columns not found")
            
        insights = {
            'total_maintenance_cost': df[cost_cols].sum().to_dict(),
            'cost_by_equipment': df.groupby(equipment_cols[0])[cost_cols[0]]
                                .agg(['sum', 'mean', 'count']).to_dict(),
            'top_maintenance_factors': self.feature_importance,
            'anomaly_count': df['is_anomaly'].sum() if 'is_anomaly' in df.columns else None
        }
        
        # Time-based analysis if date column exists
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
            monthly_data = df.set_index(date_col)[cost_cols[0]].resample('ME').sum()
            insights['monthly_trends'] = monthly_data.to_dict()
        
        return insights