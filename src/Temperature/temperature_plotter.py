from pathlib import Path
from numpy import fft

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FORECAST_HARMONICS = 10
FFT_FREQUENCY_OVERSAMPLING = 8
MIN_SEASONAL_PERIOD_DAYS = 500
MAX_SEASONAL_PERIOD_DAYS = 900


def _fourier_design(indices: np.ndarray, fundamental_frequency: float) -> np.ndarray:
    columns = [np.ones(indices.size)]
    for harmonic in range(1, FORECAST_HARMONICS + 1):
        phase = 2 * np.pi * harmonic * fundamental_frequency * indices
        columns.extend([np.cos(phase), np.sin(phase)])
    return np.column_stack(columns)


def plot_tsurf(dataframe: pd.DataFrame) -> None:
    data = (
        dataframe.groupby(["Year", "Month", "Day"], as_index=False)
        .agg(
            {
                "Fractional year": "mean",
                "Surface Temperature (°C)": "mean",
            }
        )
        .sort_values("Fractional year")
    )
    print(data)
    figure, temperature_axis = plt.subplots()

    temperature_axis.scatter(
        data["Fractional year"],
        data["Surface Temperature (°C)"],
        color="tab:red",
        marker="o",
        s=5,
        alpha=0.75,
        zorder=3,
        label="Daily mean surface temperature",
    )

    temperature_axis.set_xlabel("Year")
    temperature_axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    temperature_axis.set_ylabel("Surface Temperature (°C)", color="tab:red")
    temperature_axis.tick_params(axis="y", labelcolor="tab:red")
    temperature_axis.grid(True, axis="x", alpha=0.3)
    temperature_axis.legend(loc="best")

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
    data = (
        dataframe.groupby(["Year", "Month", "Day"], as_index=False)
        .agg(
            {
                "Fractional year": "mean",
                "Surface Temperature (°C)": "mean",
            }
        )
        .sort_values("Fractional year")
    )
    temperatures = data["Surface Temperature (°C)"].to_numpy(dtype=float)
    mean_temperature = temperatures.mean()
    fft_vals = fft.fft(temperatures - mean_temperature)
    freq = fft.fftfreq(temperatures.size, d=1.0)
    freq = fft.fftshift(freq)
    fft_vals = fft.fftshift(fft_vals)
    fig, axes = plt.subplots(2, 1)

    axes[0].plot(freq, fft_vals.real)
    axes[0].set_xlabel("Frequency (cycles/day)")
    axes[0].set_ylabel("Real part")
    axes[0].set_title("Real part of the FFT")
    axes[0].grid(True)

    axes[1].plot(freq, fft_vals.imag)
    axes[1].set_xlabel("Frequency (cycles/day)")
    axes[1].set_ylabel("Imaginary part")
    axes[1].set_title("Imaginary part of the FFT")
    axes[1].grid(True)

    last_date = pd.Timestamp(
        year=int(data["Year"].iloc[-1]),
        month=int(data["Month"].iloc[-1]),
        day=int(data["Day"].iloc[-1]),
    )
    forecast_end = pd.Timestamp("2026-12-31")
    forecast_days = (forecast_end - last_date).days
    if forecast_days < 1:
        raise ValueError("The data must end before 2026-12-31 to forecast through 2026.")
    forecast, period_days = _fourier_forecast(temperatures, forecast_days)
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        end=forecast_end,
        freq="D",
    )
    future_years = future_dates.year
    year_starts = pd.to_datetime(future_years.astype(str) + "-01-01")
    next_year_starts = pd.to_datetime((future_years + 1).astype(str) + "-01-01")
    future_fractional_years = future_years + (
        (future_dates - year_starts) / (next_year_starts - year_starts)
    )

    forecast_figure, forecast_axis = plt.subplots()
    observed_data = data.loc[data["Year"] >= 2015]
    forecast_axis.scatter(
        observed_data["Fractional year"],
        observed_data["Surface Temperature (°C)"],
        color="tab:red",
        marker="o",
        s=5,
        alpha=0.7,
        zorder=3,
        label="Observed daily mean surface temperature",
    )
    forecast_axis.plot(
        future_fractional_years,
        forecast,
        color="tab:blue",
        linestyle="--",
        linewidth=1.5,
        label=f"Fourier forecast ({FORECAST_HARMONICS} harmonics)",
    )
    forecast_axis.axvline(
        data["Fractional year"].iloc[-1],
        color="gray",
        linestyle=":",
        label="Forecast starts",
    )
    forecast_axis.set_xlabel("Year")
    forecast_axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    forecast_axis.set_ylabel("Surface Temperature (°C)")
    forecast_axis.set_title(
        f"Fourier projection through 2026 (~{period_days:.0f}-day seasonal cycle)"
    )
    forecast_axis.grid(True, axis="x", alpha=0.3)
    forecast_axis.legend(loc='upper right')

    fig.tight_layout()
    forecast_figure.tight_layout()
    plt.show()

def main():
    csv_path = Path(__file__).with_name("mars_surface_data.csv")
    surf_df = pd.read_csv(csv_path)
    plot_fft(surf_df)

if __name__ == "__main__":
    main()