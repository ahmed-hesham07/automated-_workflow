import pandas as pd
from data_ingestion import DataIngestion
from ml_analysis import EquipmentAnalysis
from report_generator import ReportGenerator
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load sample data
        logger.info("Loading sample maintenance data...")
        df = pd.read_csv('data/equipment_maintenance_sample.csv')
        df['maintenance_date'] = pd.to_datetime(df['maintenance_date'])
        
        # Initialize analysis components
        logger.info("Initializing analysis pipeline...")
        analyzer = EquipmentAnalysis()
        
        # Perform ML analysis
        logger.info("Training ML model and generating metrics...")
        metrics = analyzer.train_model(df)
        logger.info(f"Model metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
        
        # Detect anomalies
        logger.info("Detecting anomalies in maintenance patterns...")
        df_with_anomalies = analyzer.detect_anomalies(df)
        
        # Generate business insights
        logger.info("Generating business insights...")
        insights = analyzer.get_business_insights(df_with_anomalies)
        
        # Generate comprehensive report with PDF
        logger.info("Generating analysis report...")
        report_gen = ReportGenerator('reports')
        report_path = report_gen.create_report(insights, df_with_anomalies)
        
        logger.info(f"Analysis complete! Report generated at: {report_path}")
        logger.info(f"PDF report available at: {Path(report_path) / 'report.pdf'}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()