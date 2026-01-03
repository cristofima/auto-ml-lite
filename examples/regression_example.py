import pandas as pd
from sklearn.datasets import load_diabetes

from ez_automl_lite import AutoML


def test_regression():
    print("\n--- Testing Regression on Diabetes Dataset ---")

    # 1. Load data
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target_disease_progression"] = data.target

    print(f"Dataset loaded. Shape: {df.shape}")

    # 2. EDA Report
    target = "target_disease_progression"
    time_budget = 60
    aml = AutoML(target=target, time_budget=time_budget)
    print("Generating EDA report...")
    aml.eda(df, "regression_eda_report.html")

    # 3. Initialize and Train
    print(f"Starting training (target='{target}', time_budget={time_budget})...")
    aml.fit(df)

    print("\n--- Training Results ---")
    print(f"Problem Type: {aml.problem_type}")
    print(f"Best Estimator: {aml.metrics.get('best_estimator')}")
    print(f"R2 Score: {aml.metrics.get('r2_score'):.4f}")

    # 4. Generate Report
    report_name = "regression_training_report.html"
    aml.report(report_name)
    print(f"Regression report generated: {report_name}")

    # 5. Export to ONNX
    onnx_name = "regression_model.onnx"
    success = aml.export_onnx(onnx_name)
    if success:
        print(f"Model exported to ONNX: {onnx_name}")


if __name__ == "__main__":
    test_regression()
