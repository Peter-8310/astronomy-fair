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

A = 19.8
B = 118600

lower_A = A - 4.9
lower_B = B - 2500

higher_A = A + 4.9
higher_B = B + 2500

dose = (A + B / phi) * 3
lower_dose = (lower_A + lower_B / phi) * 3
higher_dose = (higher_A + higher_B / phi) * 3

fig, ax = plt.subplots(figsize=(12, 6))
ax.fill_between(
    dates,
    lower_dose,
    higher_dose,
    color="steelblue",
    alpha=0.2,
    label="Lower/upper estimates",
)
ax.plot(dates, dose, color="navy", linewidth=0.8, label="Estimated dose rate")
ax.set_title("Estimated Dose Rate (2010-2021)")
ax.set_xlabel("Date")
ax.set_ylabel("Dose rate (µSv)")
for statistic, phi_value, color in (
    ("Peak", phi.max(), "red"),
    ("Average", phi.mean(), "green"),
    ("Minimum", phi.min(), "orange"),
):
    reference_dose = (A + B / phi_value) * 3
    lower_reference_dose = (lower_A + lower_B / phi_value) * 3
    higher_reference_dose = (higher_A + higher_B / phi_value) * 3
    ax.axhspan(
        lower_reference_dose,
        higher_reference_dose,
        color=color,
        alpha=0.1,
    )
    ax.axhline(
        reference_dose,
        color=color,
        linestyle="--",
        label=(
            f"{statistic} phi ({phi_value:.1f} MV): "
            f"{reference_dose:.1f} µSv "
            f"(range {lower_reference_dose:.1f}-{higher_reference_dose:.1f} µSv)"
        ),
    )
ax.grid(True, alpha=0.3)
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
plt.show()
