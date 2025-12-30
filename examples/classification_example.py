import pandas as pd
from ez_automl_lite import AutoML
from sklearn.datasets import load_breast_cancer

def test_classification():
    print("\n--- Testing Classification on Breast Cancer Dataset ---")
    
    # 1. Load data
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    # Map to names for a more realistic test
    df['target'] = df['target'].map({0: 'malignant', 1: 'benign'})
    
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # 2. EDA Report
    target = "target"
    time_budget = 60
    aml = AutoML(target=target, time_budget=time_budget)
    print("Generating EDA report...")
    aml.eda(df, "cancer_eda_report.html")
    
    # 3. Initialize and Train
    print(f"Starting training (target='{target}', time_budget={time_budget}s)...")
    aml.fit(df)
    
    print("\n--- Training Results ---")
    print(f"Problem Type: {aml.problem_type}")
    print(f"Best Estimator: {aml.metrics.get('best_estimator')}")
    print(f"Accuracy: {aml.metrics.get('accuracy'):.4f}")
    
    # 4. Generate Report
    report_name = "cancer_training_report.html"
    aml.report(report_name)
    print(f"Classification report generated: {report_name}")
    
    # 5. Export to ONNX
    onnx_name = "cancer_model.onnx"
    success = aml.export_onnx(onnx_name)
    if success:
        print(f"Model exported to ONNX: {onnx_name}")

if __name__ == "__main__":
    test_classification()
