from pathlib import Path
from numpy import fft

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FORECAST_HARMONICS = 100
FFT_FREQUENCY_OVERSAMPLING = 8
MIN_SEASONAL_PERIOD_DAYS = 500
MAX_SEASONAL_PERIOD_DAYS = 900
FORECAST_END = pd.Timestamp("2032-12-31")
LOCATION_TEMPERATURES = (
    ("AP-1", "Surface Temperature AP-1 (°C)", "tab:blue"),
    ("AP-8", "Surface Temperature AP-8 (°C)", "tab:orange"),
    ("AP-9", "Surface Temperature AP-9 (°C)", "tab:green"),
)


def _fourier_design(indices: np.ndarray, fundamental_frequency: float) -> np.ndarray:
    columns = [np.ones(indices.size)]
    for harmonic in range(1, FORECAST_HARMONICS + 1):
        phase = 2 * np.pi * harmonic * fundamental_frequency * indices
        columns.extend([np.cos(phase), np.sin(phase)])
    return np.column_stack(columns)


def _daily_means(dataframe: pd.DataFrame) -> pd.DataFrame:
    temperature_columns = [column for _, column, _ in LOCATION_TEMPERATURES]
    required_columns = {
        "Year",
        "Month",
        "Day",
        "Fractional year",
        *temperature_columns,
    }
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing required columns: {missing}. Regenerate the CSV with "
            "the three-location data processor."
        )

    aggregations = {"Fractional year": "mean"}
    aggregations.update({column: "mean" for column in temperature_columns})
    return (
        dataframe.groupby(["Year", "Month", "Day"], as_index=False)
        .agg(aggregations)
        .sort_values("Fractional year")
    )


def plot_tsurf(dataframe: pd.DataFrame) -> None:
    data = _daily_means(dataframe)
    figure, temperature_axis = plt.subplots()

    for location, column, color in LOCATION_TEMPERATURES:
        if location != 'AP-8':
            temperature_axis.scatter(
                data["Fractional year"],
                data[column],
                s=1,
                color=color,
                label=location,
            )
        else:
            temperature_axis.scatter(
                    data["Fractional year"],
                    data[column],
                    s=1,
                    marker='x',
                    color=color,
                    label=location,
                )

    temperature_axis.set_xlabel("Year")
    temperature_axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    temperature_axis.set_ylabel("Surface Temperature (°C)")
    temperature_axis.grid(True, axis="x", alpha=0.3)
    temperature_axis.legend(title="Location", loc="best")

    figure.tight_layout()
    plt.show()


def _fourier_forecast(
    temperatures: np.ndarray, horizon_days: int
) -> tuple[np.ndarray, float]:
    temperatures = np.asarray(temperatures, dtype=float)
    if temperatures.ndim != 1 or temperatures.size < 2:
        raise ValueError("Fourier forecasting requires at least two daily values.")
    if not np.isfinite(temperatures).all():
        raise ValueError("Temperature values must all be finite.")
    if horizon_days < 1:
        raise ValueError("Forecast horizon must be at least one day.")

    sample_count = temperatures.size
    fft_size = sample_count * FFT_FREQUENCY_OVERSAMPLING
    coefficients = fft.rfft(temperatures - temperatures.mean(), n=fft_size)
    frequencies = fft.rfftfreq(fft_size, d=1.0)
    seasonal_bins = np.flatnonzero(
        (frequencies >= 1 / MAX_SEASONAL_PERIOD_DAYS)
        & (frequencies <= 1 / MIN_SEASONAL_PERIOD_DAYS)
    )
    if seasonal_bins.size == 0:
        raise ValueError("Not enough daily values to resolve a seasonal frequency.")

    fundamental_frequency = frequencies[
            seasonal_bins[np.argmax(np.abs(coefficients[seasonal_bins]))]
    ]
    period_days = 1 / fundamental_frequency
    future_indices = np.arange(sample_count, sample_count + horizon_days)
    training_design = _fourier_design(
        np.arange(sample_count), fundamental_frequency
    )
    harmonic_coefficients = np.linalg.lstsq(
        training_design, temperatures, rcond=None
    )[0]
    forecast = _fourier_design(
        future_indices, fundamental_frequency
    ) @ harmonic_coefficients

    return forecast, period_days


def plot_fft(dataframe: pd.DataFrame) -> None:
    data = _daily_means(dataframe)
    last_date = pd.Timestamp(
        year=int(data["Year"].iloc[-1]),
        month=int(data["Month"].iloc[-1]),
        day=int(data["Day"].iloc[-1]),
    )
    forecast_days = (FORECAST_END - last_date).days
    if forecast_days < 1:
        raise ValueError(
            f"The data must end before {FORECAST_END.date()} to forecast."
        )
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        end=FORECAST_END,
        freq="D",
    )
    future_years = future_dates.year
    year_starts = pd.to_datetime(future_years.astype(str) + "-01-01")
    next_year_starts = pd.to_datetime((future_years + 1).astype(str) + "-01-01")
    future_fractional_years = future_years + (
        (future_dates - year_starts) / (next_year_starts - year_starts)
    )

    forecast_figure, forecast_axis = plt.subplots(3, 1)
    for i, (location, column, color) in enumerate(LOCATION_TEMPERATURES):
        temperatures = data[column].to_numpy(dtype=float)
        mean_temperature = temperatures.mean()
        fft_vals = fft.fft(temperatures - mean_temperature)
        freq = fft.fftfreq(temperatures.size, d=1.0)
        freq = fft.fftshift(freq)
        fft_vals = fft.fftshift(fft_vals)

        forecast, period_days = _fourier_forecast(temperatures, forecast_days)
        
        forecast_axis[i].scatter(
            data["Fractional year"],
            temperatures,
            color=color,
            marker="o",
            s=5,
            alpha=0.7,
            zorder=3,
            label=f"{location} observed daily mean",
        )
        forecast_axis[i].plot(
            future_fractional_years,
            forecast,
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"{location} Fourier prediction",
        )
        forecast_axis[i].axvline(
            data["Fractional year"].iloc[-1],
            color="gray",
            linestyle=":",
            label="Forecast starts",
        )
        forecast_axis[i].ticklabel_format(axis="x", style="plain", useOffset=False)
        forecast_axis[i].set_title(
            f"{location} Surface temperature projection through {FORECAST_END.year} "
            f"(~{period_days:.0f}-day seasonal cycle)"
        )
        forecast_axis[i].grid(True, axis="x", alpha=0.3)
        forecast_axis[i].legend(loc="upper right")

    forecast_axis[2].set_xlabel("Year")
    forecast_axis[1].set_ylabel("Surface Temperature (°C)")
    forecast_figure.tight_layout()
    plt.show()

def main():
    csv_path = Path(__file__).with_name("mars_surface_data.csv")
    surf_df = pd.read_csv(csv_path)
    plot_fft(surf_df)

if __name__ == "__main__":
    main()