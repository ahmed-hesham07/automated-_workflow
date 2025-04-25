import unittest
import pandas as pd
import numpy as np
from src.data_ingestion import DataIngestion
from src.ml_analysis import EquipmentAnalysis
from src.report_generator import ReportGenerator
from pathlib import Path
import tempfile
import shutil

class TestEquipmentAnalysis(unittest.TestCase):
    def setUp(self):
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        n_samples = len(dates)
        
        self.test_data = pd.DataFrame({
            'maintenance_date': dates,
            'equipment_id': np.random.choice(['EQ001', 'EQ002', 'EQ003'], n_samples),
            'maintenance_cost': np.random.normal(1000, 200, n_samples),
            'repair_hours': np.random.normal(5, 1, n_samples),
            'equipment_type': np.random.choice(['Type A', 'Type B', 'Type C'], n_samples),
            'failure_type': np.random.choice(['Mechanical', 'Electrical', 'Hydraulic'], n_samples)
        })
        
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_data_validation(self):
        ingestion = DataIngestion()
        self.assertTrue(ingestion.validate_data(self.test_data))
        
        # Test with missing required columns
        invalid_data = self.test_data.drop(columns=['maintenance_cost'])
        self.assertFalse(ingestion.validate_data(invalid_data))
        
    def test_ml_analysis(self):
        analyzer = EquipmentAnalysis()
        metrics = analyzer.train_model(self.test_data)
        
        # Check if metrics are generated
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('feature_importance', metrics)
        
        # Check anomaly detection
        df_with_anomalies = analyzer.detect_anomalies(self.test_data)
        self.assertIn('is_anomaly', df_with_anomalies.columns)
        
    def test_report_generation(self):
        analyzer = EquipmentAnalysis()
        metrics = analyzer.train_model(self.test_data)
        df_with_anomalies = analyzer.detect_anomalies(self.test_data)
        insights = analyzer.get_business_insights(df_with_anomalies)
        
        report_gen = ReportGenerator(str(self.temp_dir))
        report_path = report_gen.create_report(insights, df_with_anomalies)
        
        # Check if report files are generated
        report_dir = Path(report_path)
        self.assertTrue(report_dir.exists())
        self.assertTrue((report_dir / 'report.html').exists())
        self.assertTrue((report_dir / 'insights.json').exists())

if __name__ == '__main__':
    unittest.main()