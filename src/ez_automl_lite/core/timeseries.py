"""
Automated Time Series Forecasting Module.
"""

import time
import uuid
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ez_automl_lite.reports.eda import generate_eda_report
from ez_automl_lite.reports.timeseries_report import generate_timeseries_report


try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    # Avoid division by zero
    mask = denominator != 0
    smape_values = np.zeros_like(diff)
    smape_values[mask] = diff[mask] / denominator[mask]
    return float(100.0 * np.mean(smape_values))


class AutoTimeSeries:
    """
    Automated Time Series Forecasting using ARIMA/SARIMA or Prophet.
    """

    def __init__(
        self,
        time_column: str,
        target_column: str,
        frequency: str = "D",
        forecast_horizon: int = 30,
        random_state: int = 42,
        scaling: str = "absmax",
        holidays_mode: str | None = None,
        job_id: str | None = None,
    ):
        """
        Initialize AutoTimeSeries.

        Args:
            time_column: Name of the datetime column
            target_column: Name of the target column to forecast
            frequency: Frequency of the time series ('D' for daily, 'W' for weekly, 'M' for monthly, etc.)
            forecast_horizon: Number of time steps to forecast ahead
            random_state: Random seed for reproducibility
            scaling: Scaling method for Prophet ('absmax' or 'minmax'). Default: 'absmax'
            holidays_mode: Separate mode for holidays in Prophet ('additive' or 'multiplicative'). Default: None (uses seasonality_mode)
            job_id: Optional job identifier for tracking
        """
        self.time_column = time_column
        self.target_column = target_column
        self.frequency = frequency
        self.forecast_horizon = forecast_horizon
        self.random_state = random_state
        self.scaling = scaling
        self.holidays_mode = holidays_mode
        self.job_id = job_id or str(uuid.uuid4())
        self.model = None
        self.metrics = {}
        self.dataset_info = {}
        self.algorithm_name = "ARIMA"
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_lower = None
        self.y_pred_upper = None
        self.forecast_values = None
        self.forecast_dates = None
        self.forecast_lower = None
        self.forecast_upper = None
        self.time_index = None
        self.decomposition = None

    def _check_stationarity(self, series: pd.Series) -> dict[str, Any]:
        """Check if time series is stationary using Augmented Dickey-Fuller test."""
        if not STATSMODELS_AVAILABLE:
            return {"is_stationary": None, "adf_statistic": None, "p_value": None}

        try:
            result = adfuller(series.dropna())
            return {
                "is_stationary": result[1] < 0.05,  # p-value < 0.05
                "adf_statistic": float(result[0]),
                "p_value": float(result[1]),
                "critical_values": {k: float(v) for k, v in result[4].items()},
            }
        except Exception as e:
            print(f"Warning: Could not perform stationarity test: {e}")
            return {"is_stationary": None, "adf_statistic": None, "p_value": None}

    def _decompose_series(self, series: pd.Series, model: str = "additive") -> dict[str, Any]:
        """Decompose time series into trend, seasonal, and residual components."""
        if not STATSMODELS_AVAILABLE:
            return {}

        try:
            # Need at least 2 full cycles for decomposition
            period = self._infer_period()
            if len(series) < 2 * period:
                print(
                    f"Warning: Time series too short for decomposition (need at least {2 * period} points)"
                )
                return {}

            decomposition = seasonal_decompose(
                series, model=model, period=period, extrapolate_trend="freq"
            )

            return {
                "trend": decomposition.trend,
                "seasonal": decomposition.seasonal,
                "residual": decomposition.resid,
            }
        except Exception as e:
            print(f"Warning: Could not decompose time series: {e}")
            return {}

    def _infer_period(self) -> int:
        """Infer seasonality period from frequency."""
        freq_map = {
            "D": 7,  # Daily -> weekly seasonality
            "W": 52,  # Weekly -> yearly seasonality
            "M": 12,  # Monthly -> yearly seasonality
            "Q": 4,  # Quarterly -> yearly seasonality
            "H": 24,  # Hourly -> daily seasonality
        }
        return freq_map.get(self.frequency, 12)

    def _train_arima(self, y_train: pd.Series, y_test: pd.Series) -> tuple[Any, np.ndarray]:
        """Train ARIMA/SARIMA model."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels is required for ARIMA. Install with: pip install 'ez-automl-lite[timeseries]'"
            )

        # Simple auto-ARIMA: try a few common configurations
        best_model = None
        best_aic = float("inf")
        best_order = None

        # Common ARIMA orders to try
        orders = [
            (1, 1, 1),
            (2, 1, 2),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 0),
        ]

        print("Searching for best ARIMA configuration...")
        for order in orders:
            try:
                model = SARIMAX(
                    y_train, order=order, enforce_stationarity=False, enforce_invertibility=False
                )
                fitted = model.fit(disp=False)
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model = fitted
                    best_order = order
            except Exception:  # nosec B112
                continue

        if best_model is None:
            raise ValueError("Could not fit any ARIMA model")

        print(f"Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")

        # Forecast on test set
        forecast = best_model.forecast(steps=len(y_test))

        return best_model, np.array(forecast)

    def _train_prophet(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> tuple[Any, np.ndarray]:
        """Train Prophet model."""
        if not PROPHET_AVAILABLE:
            raise ImportError(
                "prophet is required for Prophet algorithm. Install with: pip install 'ez-automl-lite[timeseries]'"
            )

        # Prophet requires specific column names
        prophet_train = pd.DataFrame(
            {
                "ds": df_train[self.time_column],
                "y": df_train[self.target_column],
            }
        )

        # Use Prophet 1.1.5+ features: scaling and holidays_mode
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            scaling=self.scaling,  # 'absmax' or 'minmax' (new in 1.1.5)
            holidays_mode=self.holidays_mode,  # Optional separate holidays mode (new in 1.1.5)
            interval_width=0.95,  # 95% confidence intervals
        )

        # Suppress Prophet's verbose output
        import logging

        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        try:
            model.fit(prophet_train)
        except Exception as e:
            print(f"Warning: Prophet fit failed with error: {e}. Retrying with default settings...")
            # Fallback to default settings if custom parameters fail
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
            )
            model.fit(prophet_train)

        # Create future dataframe for predictions
        future = pd.DataFrame({"ds": df_test[self.time_column]})
        forecast = model.predict(future)

        # Store confidence intervals
        self.y_pred_lower = forecast["yhat_lower"].values
        self.y_pred_upper = forecast["yhat_upper"].values

        return model, forecast["yhat"].values

    def fit(  # noqa: PLR0912, PLR0915
        self, df: pd.DataFrame, algorithm: str = "auto", test_size: float = 0.2
    ) -> "AutoTimeSeries":
        """
        Train the time series forecasting model.

        Args:
            df: DataFrame with time series data
            algorithm: 'auto', 'arima', or 'prophet'
            test_size: Fraction of data to use for testing
        """
        print(
            f"Starting time series training (Algorithm: {algorithm}, Horizon: {self.forecast_horizon})..."
        )

        # Validate required columns
        if self.time_column not in df.columns:
            raise ValueError(f"Time column '{self.time_column}' not found in DataFrame")
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")

        # Sort by time and set index
        df_sorted = df.copy()
        df_sorted[self.time_column] = pd.to_datetime(df_sorted[self.time_column])
        df_sorted = df_sorted.sort_values(by=self.time_column).reset_index(drop=True)
        self.time_index = df_sorted[self.time_column]

        # Get target series
        y = df_sorted[self.target_column]

        # Store dataset info
        min_date = df_sorted[self.time_column].min()
        max_date = df_sorted[self.time_column].max()

        # Format dates without time if frequency is daily
        if self.frequency.upper() in [
            "D",
            "B",
            "W",
            "W-MON",
            "W-TUE",
            "W-WED",
            "W-THU",
            "W-FRI",
            "W-SAT",
            "W-SUN",
            "M",
            "MS",
            "Q",
            "QS",
            "Y",
            "YS",
            "A",
            "AS",
        ]:
            date_range_str = f"{min_date.date()} to {max_date.date()}"
        else:
            date_range_str = f"{min_date} to {max_date}"

        self.dataset_info = {
            "total_samples": len(df_sorted),
            "date_range": date_range_str,
            "frequency": self.frequency,
            "target_column": self.target_column,
        }

        # Check stationarity
        stationarity = self._check_stationarity(y)
        self.metrics["stationarity"] = stationarity

        # Decompose series
        self.decomposition = self._decompose_series(y)

        # Train/test split (temporal split - no shuffle!)
        split_idx = int(len(df_sorted) * (1 - test_size))
        df_train = df_sorted.iloc[:split_idx].copy()
        df_test = df_sorted.iloc[split_idx:].copy()

        y_train = df_train[self.target_column]
        y_test = df_test[self.target_column]

        self.y_train = y_train
        self.y_test = y_test

        # Auto-select algorithm
        if algorithm == "auto":
            # Default to ARIMA if statsmodels available, otherwise Prophet
            if STATSMODELS_AVAILABLE:
                algorithm = "arima"
            elif PROPHET_AVAILABLE:
                algorithm = "prophet"
            else:
                raise ImportError(
                    "No time series library available. Install with: pip install 'ez-automl-lite[timeseries]'"
                )

        self.algorithm_name = algorithm.upper()

        start_time = time.time()

        # Train model
        if algorithm == "arima":
            self.model, self.y_pred = self._train_arima(y_train, y_test)
        elif algorithm == "prophet":
            self.model, self.y_pred = self._train_prophet(df_train, df_test)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        execution_time = time.time() - start_time

        # Calculate metrics
        mae = mean_absolute_error(y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, self.y_pred))
        smape = calculate_smape(y_test.values, self.y_pred)

        # Calculate MAPE (avoid division by zero)
        mask = y_test != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - self.y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = None

        self.metrics.update(
            {
                "mae": float(mae),
                "rmse": float(rmse),
                "smape": float(smape),
                "mape": float(mape) if mape is not None else None,
                "execution_time": execution_time,
                "algorithm": self.algorithm_name,
                "forecast_horizon": self.forecast_horizon,
                "train_size": len(y_train),
                "test_size": len(y_test),
                "job_id": self.job_id,
            }
        )

        print(
            f"Training completed in {execution_time:.2f}s | MAE: {mae:.4f} | RMSE: {rmse:.4f} | sMAPE: {smape:.2f}%"
        )

        return self

    def predict(self, horizon: int | None = None) -> np.ndarray:
        """
        Generate forecasts for future time steps.

        Args:
            horizon: Number of steps to forecast (default: self.forecast_horizon)

        Returns:
            Array of forecasted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        horizon = horizon or self.forecast_horizon

        if self.algorithm_name == "ARIMA":
            forecast = self.model.forecast(steps=horizon)
            self.forecast_values = np.array(forecast)
            # ARIMA: Get confidence intervals if available
            try:
                forecast_result = self.model.get_forecast(steps=horizon)
                conf_int = forecast_result.conf_int()
                self.forecast_lower = conf_int.iloc[:, 0].values
                self.forecast_upper = conf_int.iloc[:, 1].values
            except Exception:
                self.forecast_lower = None
                self.forecast_upper = None
        elif self.algorithm_name == "PROPHET":
            # Create future dates
            last_date = self.time_index.iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=self.frequency)[
                1:
            ]
            future = pd.DataFrame({"ds": future_dates})
            forecast = self.model.predict(future)
            self.forecast_values = forecast["yhat"].values
            self.forecast_lower = forecast["yhat_lower"].values
            self.forecast_upper = forecast["yhat_upper"].values
        else:
            raise ValueError(f"Prediction not implemented for {self.algorithm_name}")

        # Store forecast dates
        last_date = self.time_index.iloc[-1]
        self.forecast_dates = pd.date_range(
            start=last_date, periods=horizon + 1, freq=self.frequency
        )[1:]

        return self.forecast_values

    def report(self, output_path: str = "timeseries_training_report.html") -> None:
        """Generate time series training report."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        generate_timeseries_report(
            output_path=output_path,
            job_id=self.job_id,
            metrics=self.metrics,
            dataset_info=self.dataset_info,
            y_train=self.y_train,
            y_test=self.y_test,
            y_pred=self.y_pred,
            y_pred_lower=self.y_pred_lower,
            y_pred_upper=self.y_pred_upper,
            time_index=self.time_index,
            forecast_values=self.forecast_values,
            forecast_dates=self.forecast_dates,
            forecast_lower=self.forecast_lower,
            forecast_upper=self.forecast_upper,
            decomposition=self.decomposition,
        )

    def eda(self, df: pd.DataFrame, output_path: str = "timeseries_eda_report.html") -> None:
        """Generate EDA report for time series data."""
        generate_eda_report(
            df, target_column=self.target_column, output_path=output_path, task_type="timeseries"
        )
