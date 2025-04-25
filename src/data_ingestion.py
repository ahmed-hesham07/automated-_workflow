from typing import Dict, Any
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

class DataIngestion:
    def __init__(self):
        load_dotenv()
        self.db_connection = os.getenv('DATABASE_URL')
        
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute any SQL query and return results as a DataFrame"""
        engine = create_engine(self.db_connection)
        return pd.read_sql(query, engine)
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate if the dataframe has required columns for equipment analysis"""
        df_columns = df.columns.str.lower()
        
        # Required column patterns
        patterns = {
            'cost': ['cost', 'price', 'amount', 'value'],
            'date': ['date', 'timestamp'],
            'equipment': ['equipment', 'type', 'status', 'category']
        }
        
        # Find matching columns for each pattern
        matches = {
            key: [col for col in df_columns 
                 if any(pattern in col for pattern in patterns[key])]
            for key in patterns
        }
        
        # Check if we have at least one column for each required type
        return all(len(cols) > 0 for cols in matches.values())