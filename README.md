# Equipment Maintenance Analysis Pipeline

An automated workflow for analyzing equipment maintenance data using ML/AI techniques. This system is designed to work with any SQL query result containing equipment maintenance data, providing standardized analysis and reporting.

## Features

- Dynamic SQL data ingestion with automatic column detection
- Automated ML analysis including:
  - Cost prediction modeling
  - Anomaly detection in maintenance patterns
  - Feature importance analysis
- Business insights generation
- Standardized HTML report generation with visualizations
- Flexible data validation that adapts to your column names

## Setup

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the configuration template and update with your database details:
   ```bash
   cp .env.template .env
   ```
   Edit `.env` with your database connection details.

## Usage

Run the analysis pipeline with your SQL query:

```bash
python src/main.py --query "YOUR_SQL_QUERY_HERE"
```

Example query:
```sql
SELECT 
    equipment_id,
    maintenance_date,
    maintenance_cost,
    equipment_type,
    repair_hours,
    failure_type
FROM maintenance_records
WHERE maintenance_date >= '2024-01-01'
```

The system will automatically:
1. Validate the data structure
2. Train an ML model for cost prediction
3. Detect maintenance anomalies
4. Generate business insights
5. Create an HTML report with visualizations

## Required Data Structure

Your SQL query should return columns that include:
- At least one cost-related column (containing terms like 'cost', 'price', 'amount')
- At least one date column (containing terms like 'date', 'timestamp')
- At least one categorical column for equipment identification (containing terms like 'equipment', 'type', 'status')

The system will automatically detect and use appropriate columns based on their names and data types.

## Output

The analysis generates a report directory containing:
- HTML report with interactive visualizations
- JSON file with detailed insights
- Visualization plots as PNG files

Reports are saved in the `reports` directory by default (configurable via --output-dir).