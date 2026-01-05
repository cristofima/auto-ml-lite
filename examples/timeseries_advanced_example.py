"""
Advanced Time Series Forecasting Example using Prophet 1.2.1+ features.
This example demonstrates the new scaling and holidays_mode parameters.
"""

import numpy as np
import pandas as pd

from ez_automl_lite import AutoTimeSeries


def main():  # noqa: PLR0915
    print("=" * 70)
    print("Advanced Time Series Forecasting Example")
    print("=" * 70)

    # Generate synthetic time series data with more complex patterns
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=365, freq="D")

    # Components
    trend = np.linspace(50, 150, 365)  # Upward trend
    seasonality = 30 * np.sin(np.linspace(0, 4 * np.pi, 365))  # Strong yearly seasonality
    weekly_pattern = 10 * np.sin(np.linspace(0, 52 * np.pi, 365))  # Weekly pattern
    noise = np.random.normal(0, 8, 365)

    values = trend + seasonality + weekly_pattern + noise

    df = pd.DataFrame({"date": dates, "value": values})

    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")

    # ========================================================================
    # Example 1: Default scaling (absmax)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 1: Default Scaling (absmax)")
    print("=" * 70)

    automl1 = AutoTimeSeries(
        time_column="date",
        target_column="value",
        frequency="D",
        forecast_horizon=30,
        random_state=42,
        scaling="absmax",  # Default: scales by maximum absolute value
    )

    automl1.fit(df, algorithm="prophet", test_size=0.2)

    print(f"\nAlgorithm: {automl1.metrics['algorithm']}")
    print(f"MAE: {automl1.metrics['mae']:.4f}")
    print(f"RMSE: {automl1.metrics['rmse']:.4f}")
    print(f"sMAPE: {automl1.metrics['smape']:.2f}%")
    print(f"Training Time: {automl1.metrics['execution_time']:.2f}s")

    forecast1 = automl1.predict(horizon=30)
    print("\nForecast statistics (absmax scaling):")
    print(f"  Mean: {forecast1.mean():.2f}")
    print(f"  Std: {forecast1.std():.2f}")

    automl1.report("timeseries_absmax_report.html")
    print("[OK] Report saved: timeseries_absmax_report.html")

    # ========================================================================
    # Example 2: MinMax scaling
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: MinMax Scaling")
    print("=" * 70)

    automl2 = AutoTimeSeries(
        time_column="date",
        target_column="value",
        frequency="D",
        forecast_horizon=30,
        random_state=42,
        scaling="minmax",  # Scales between 0 and 1
    )

    automl2.fit(df, algorithm="prophet", test_size=0.2)

    print(f"\nAlgorithm: {automl2.metrics['algorithm']}")
    print(f"MAE: {automl2.metrics['mae']:.4f}")
    print(f"RMSE: {automl2.metrics['rmse']:.4f}")
    print(f"sMAPE: {automl2.metrics['smape']:.2f}%")
    print(f"Training Time: {automl2.metrics['execution_time']:.2f}s")

    forecast2 = automl2.predict(horizon=30)
    print("\nForecast statistics (minmax scaling):")
    print(f"  Mean: {forecast2.mean():.2f}")
    print(f"  Std: {forecast2.std():.2f}")

    automl2.report("timeseries_minmax_report.html")
    print("[OK] Report saved: timeseries_minmax_report.html")

    # ========================================================================
    # Example 3: Custom holidays mode
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 3: Custom Holidays Mode")
    print("=" * 70)

    automl3 = AutoTimeSeries(
        time_column="date",
        target_column="value",
        frequency="D",
        forecast_horizon=30,
        random_state=42,
        scaling="absmax",
        holidays_mode="multiplicative",  # Separate mode for holidays
    )

    automl3.fit(df, algorithm="prophet", test_size=0.2)

    print(f"\nAlgorithm: {automl3.metrics['algorithm']}")
    print(f"MAE: {automl3.metrics['mae']:.4f}")
    print(f"RMSE: {automl3.metrics['rmse']:.4f}")
    print(f"sMAPE: {automl3.metrics['smape']:.2f}%")
    print(f"Training Time: {automl3.metrics['execution_time']:.2f}s")

    forecast3 = automl3.predict(horizon=30)
    print("\nForecast statistics (holidays_mode=multiplicative):")
    print(f"  Mean: {forecast3.mean():.2f}")
    print(f"  Std: {forecast3.std():.2f}")

    # Display confidence intervals
    if automl3.forecast_lower is not None:
        print("\n" + "-" * 70)
        print("95% Confidence Intervals (First 5 days):")
        print("-" * 70)
        for i in range(5):
            lower = automl3.forecast_lower[i]
            upper = automl3.forecast_upper[i]
            width = upper - lower
            print(f"Day {i+1}: {forecast3[i]:.2f} ± {width/2:.2f} (CI: [{lower:.2f}, {upper:.2f}])")

    automl3.report("timeseries_holidays_report.html")
    print("[OK] Report saved: timeseries_holidays_report.html")

    # ========================================================================
    # Comparison Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Performance Comparison")
    print("=" * 70)

    comparison = pd.DataFrame(
        {
            "Configuration": ["absmax", "minmax", "holidays_mode"],
            "MAE": [automl1.metrics["mae"], automl2.metrics["mae"], automl3.metrics["mae"]],
            "RMSE": [automl1.metrics["rmse"], automl2.metrics["rmse"], automl3.metrics["rmse"]],
            "sMAPE (%)": [
                automl1.metrics["smape"],
                automl2.metrics["smape"],
                automl3.metrics["smape"],
            ],
            "Time (s)": [
                automl1.metrics["execution_time"],
                automl2.metrics["execution_time"],
                automl3.metrics["execution_time"],
            ],
        }
    )

    print(comparison.to_string(index=False))

    print("\n" + "=" * 70)
    print("Advanced Example Completed Successfully!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  - timeseries_absmax_report.html")
    print("  - timeseries_minmax_report.html")
    print("  - timeseries_holidays_report.html")


if __name__ == "__main__":
    main()
