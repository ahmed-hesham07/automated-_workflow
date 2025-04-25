import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path
from jinja2 import Template
from weasyprint import HTML

class ReportGenerator:
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def _convert_timestamps(self, obj):
        """Convert pandas Timestamps to ISO format strings"""
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {str(k): self._convert_timestamps(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_timestamps(item) for item in obj]
        return obj
        
    def create_report(self, insights: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generate a comprehensive report from the analysis insights"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f'equipment_analysis_report_{timestamp}'
        report_path.mkdir(exist_ok=True)
        
        # Convert timestamps before JSON serialization
        processed_insights = self._convert_timestamps(insights)
        
        # Save insights as JSON
        with open(report_path / 'insights.json', 'w') as f:
            json.dump(processed_insights, f, indent=4, default=str)
            
        # Generate visualizations
        self._create_visualizations(insights, df, report_path)
        
        # Generate HTML report
        html_report = self._generate_html_report(insights, df, report_path)
        html_path = report_path / 'report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
            
        # Convert to PDF
        pdf_path = report_path / 'report.pdf'
        HTML(string=html_report).write_pdf(pdf_path)
            
        return str(report_path)
        
    def _create_visualizations(self, insights: Dict[str, Any], df: pd.DataFrame, 
                             report_path: Path) -> None:
        """Create comprehensive visualizations for the report"""
        plt.style.use('default')
        
        # Cost trends over time
        if 'monthly_trends' in insights:
            plt.figure(figsize=(12, 6))
            monthly_costs = pd.Series({pd.Timestamp(k): v 
                                     for k, v in insights['monthly_trends'].items()})
            monthly_costs.plot(kind='line', title='Monthly Maintenance Costs')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(report_path / 'monthly_trends.png')
            plt.close()
        
        # Equipment cost distribution
        if 'cost_by_equipment' in insights:
            plt.figure(figsize=(12, 6))
            cost_data = pd.DataFrame(insights['cost_by_equipment'])
            plt.bar(range(len(cost_data)), cost_data['sum'])
            plt.xticks(range(len(cost_data)), cost_data.index, rotation=45)
            plt.title('Cost Distribution by Equipment')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(report_path / 'cost_distribution.png')
            plt.close()
            
        # Feature importance plot
        if 'top_maintenance_factors' in insights:
            plt.figure(figsize=(10, 6))
            features = pd.Series(insights['top_maintenance_factors'])
            features.sort_values(ascending=True).plot(kind='barh')
            plt.title('Feature Importance in Maintenance Prediction')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(report_path / 'feature_importance.png')
            plt.close()
            
    def _generate_html_report(self, insights: Dict[str, Any], df: pd.DataFrame, report_path: Path) -> str:
        """Generate comprehensive HTML report with insights and visualizations"""
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Equipment Maintenance Analysis Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding-top: 60px; }
                .ai-insight {
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    border-left: 4px solid #007bff;
                    padding: 15px;
                    margin-bottom: 15px;
                    border-radius: 4px;
                }
                .ai-insight.high-impact { border-left-color: #dc3545; }
                .ai-insight.medium-impact { border-left-color: #ffc107; }
                .pattern-card {
                    background: white;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .section { margin-bottom: 40px; padding: 20px; }
                img { max-width: 100%; height: auto; margin: 20px 0; }
            </style>
        </head>
        <body>
            <nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
                <div class="container-fluid">
                    <a class="navbar-brand" href="#">Equipment Maintenance Analysis</a>
                    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                        <span class="navbar-toggler-icon"></span>
                    </button>
                    <div class="collapse navbar-collapse" id="navbarNav">
                        <ul class="navbar-nav">
                            <li class="nav-item"><a class="nav-link" href="#overview">Overview</a></li>
                            <li class="nav-item"><a class="nav-link" href="#costs">Cost Analysis</a></li>
                            <li class="nav-item"><a class="nav-link" href="#patterns">Maintenance Patterns</a></li>
                            <li class="nav-item"><a class="nav-link" href="#insights">AI Insights</a></li>
                        </ul>
                    </div>
                </div>
            </nav>

            <div class="container">
                <section id="overview" class="section">
                    <h2>Overview</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-body">
                                    <h5 class="card-title">Total Maintenance Cost</h5>
                                    <p class="card-text display-4">${total_cost:,.2f}</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-body">
                                    <h5 class="card-title">Anomalies Detected</h5>
                                    <p class="card-text display-4">{anomaly_count}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                <section id="costs" class="section">
                    <h2>Cost Analysis</h2>
                    <div class="visualization">
                        <img src="monthly_trends.png" alt="Monthly Cost Trends" class="img-fluid">
                    </div>
                    <div class="visualization">
                        <img src="cost_distribution.png" alt="Cost Distribution by Equipment" class="img-fluid">
                    </div>
                </section>

                <section id="patterns" class="section">
                    <h2>Maintenance Patterns</h2>
                    <div class="visualization">
                        <img src="feature_importance.png" alt="Feature Importance in Maintenance" class="img-fluid">
                    </div>
                </section>

                <section id="insights" class="section">
                    <h2>AI-Driven Insights</h2>
                    {ai_insights}
                </section>
            </div>

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        '''
        
        # Generate AI insights HTML
        ai_insights_html = self._generate_ai_insights(insights, df)
        
        # Calculate total cost
        total_cost = sum(insights.get('total_maintenance_cost', {}).values())
        anomaly_count = insights.get('anomaly_count', 'N/A')
        
        # Render template
        template = Template(html_template)
        return template.render(
            total_cost=total_cost,
            anomaly_count=anomaly_count,
            ai_insights=ai_insights_html
        )
    
    def _generate_ai_insights(self, insights: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generate AI insights section HTML"""
        ai_insights = []
        
        # Cost-related insights
        if 'cost_by_equipment' in insights:
            cost_data = pd.DataFrame(insights['cost_by_equipment'])
            high_cost_equipment = cost_data.nlargest(3, 'sum').index.tolist()
            ai_insights.append({
                'title': 'Cost Optimization Opportunities',
                'finding': f'Equipment {", ".join(high_cost_equipment)} have significantly higher costs',
                'details': 'These items may present opportunities for cost optimization',
                'impact': 'high'
            })
            
        # Anomaly-related insights
        if 'anomaly_count' in insights and insights['anomaly_count'] > 0:
            ai_insights.append({
                'title': 'Maintenance Anomalies Detected',
                'finding': f'Detected {insights["anomaly_count"]} unusual maintenance events',
                'details': 'These events may require further investigation',
                'impact': 'medium'
            })
            
        # Feature importance insights
        if 'feature_importance' in insights:
            features = pd.Series(insights['feature_importance'])
            top_features = features.nlargest(3)
            ai_insights.append({
                'title': 'Key Maintenance Factors',
                'finding': f'Top factors affecting maintenance: {", ".join(top_features.index)}',
                'details': 'These factors have the strongest influence on maintenance patterns',
                'impact': 'medium'
            })
        
        # Generate HTML for insights
        insights_html = []
        for insight in ai_insights:
            insights_html.append(f'''
                <div class="ai-insight {insight['impact']}-impact">
                    <h4>{insight['title']}</h4>
                    <p><strong>Finding:</strong> {insight['finding']}</p>
                    <p><strong>Details:</strong> {insight['details']}</p>
                    <span class="badge bg-{'danger' if insight['impact'] == 'high' else 'warning'}">
                        {insight['impact'].title()} Impact
                    </span>
                </div>
            ''')
        
        return '\n'.join(insights_html)