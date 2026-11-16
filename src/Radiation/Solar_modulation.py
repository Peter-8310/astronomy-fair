import pandas as pd
import matplotlib.pyplot as plt

dataframe:pd.DataFrame = pd.read_csv("Phi_daily_1964-2021_Vaisanen23.csv", sep=';')
dataframe = dataframe[dataframe["Year"].between(2010, 2021)]

dates = pd.to_datetime(
    dataframe[["Year", "Month", "Day"]].rename(
        columns={"Year": "year", "Month": "month", "Day": "day"}
    )
)
phi = dataframe["Modulation potential [MV]"]
phi_peak = phi.max()
phi_average = phi.mean()
phi_median = phi.median()
phi_minimum = phi.min()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dates, phi, linewidth=0.8)
ax.axhline(phi_peak, color="red", linestyle="--", label=f"Peak: {phi_peak:.1f} MV")
ax.axhline(phi_average, color="green", linestyle="--", label=f"Mean: {phi_average:.1f} MV")
ax.axhline(phi_median, color="purple", linestyle=":", label=f"Median: {phi_median:.1f} MV")
ax.axhline(phi_minimum, color="blue", linestyle="--", label=f"Minimum: {phi_minimum:.1f} MV")
ax.set_title("Daily Modulation Potential (2010-2021)")
ax.set_xlabel("Date")
ax.set_ylabel("Modulation potential [MV]")
ax.grid(True, alpha=0.3)
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
plt.show()
