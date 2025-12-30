import pandas as pd
import numpy as np
from auto_ml_lite import AutoAnomaly
from sklearn.datasets import load_wine

def test_anomaly_detection():
    print("\n--- Testing Automated Anomaly Detection on Wine Dataset ---")
    
    # 1. Load data
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # 2. Inject some anomalies (outliers)
    print("Injecting artificial anomalies...")
    anomalies = []
    for _ in range(5):
        # Create a row with extreme values (10x the max of some features)
        extreme_row = df.iloc[0].copy()
        extreme_row['alcohol'] = df['alcohol'].max() * 5
        extreme_row['malic_acid'] = df['malic_acid'].max() * 5
        anomalies.append(extreme_row)
    
    df = pd.concat([df, pd.DataFrame(anomalies)], ignore_index=True)
    print(f"Dataset ready. Shape: {df.shape} (including 5 artificial anomalies)")
    
    # 3. Initialize and Train
    aa = AutoAnomaly(contamination=0.05) # Assume 5% anomalies
    print(f"Job ID: {aa.job_id}")
    
    aa.fit(df)
    
    print("\n--- Anomaly Detection Results ---")
    print(f"Anomalies found: {aa.metrics.get('anomaly_count')}")
    print(f"Anomaly Percentage: {aa.metrics.get('anomaly_percentage'):.2f}%")
    
    # 4. Generate Report
    report_name = "wine_anomaly_report.html"
    aa.report(report_name)
    print(f"Anomaly report generated: {report_name}")
    
    # 5. Predict
    # The last 5 rows should be detected as -1
    predictions = aa.predict(df.tail(10))
    print(f"Last 10 rows predictions (1=Normal, -1=Anomaly):")
    print(predictions)

if __name__ == "__main__":
    test_anomaly_detection()
