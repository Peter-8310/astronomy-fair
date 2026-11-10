import pandas as pd
import matplotlib.pyplot as plt
from typing import cast


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

dose_dataframe = pd.DataFrame({
        "Year":dataframe['Year'],
        "Fractional year":dataframe['Fractional year'], 
        "dose":dose, "lower_dose":lower_dose, 
        "higher_dose":higher_dose
        }
    )

print(dose_dataframe)

def rieman_sum(df:pd.DataFrame, year:int):
    
    year_dose = df[df['Year'].between(year, year)]
    
    net_dose = 0.0
    net_dose_lower = 0.0
    net_dose_higher = 0.0
    
    for row in year_dose.itertuples():
        
        dose = float(cast(float, row.dose))
        lower_dose = float(cast(float, row.lower_dose))
        higher_dose = float(cast(float, row.higher_dose))
    
        net_dose += dose
        net_dose_higher += higher_dose
        net_dose_lower += lower_dose
    
    return net_dose, net_dose_higher, net_dose_lower


yr_dose = []
high_yr_dose = []
low_yr_dose= []

for i in range(2010, 2022):
    a, b, c = rieman_sum(dose_dataframe, i)
    yr_dose.append(a/1000)
    high_yr_dose.append(b/1000)
    low_yr_dose.append(c/1000)
    
yr_df = pd.DataFrame({"Year":range(2010, 2022), "Low yearly dose (mSv)":low_yr_dose, "Yearly dose (mSv)":yr_dose, "High yearly dose (mSv)":high_yr_dose})
daily_stats = dose_dataframe.groupby("Year")["dose"].agg(
    average="mean",
    minimum="min",
    maximum="max",
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(
    yr_df["Year"],
    yr_df["Low yearly dose (mSv)"],
    yr_df["High yearly dose (mSv)"],
    color="steelblue",
    alpha=0.2,
    label="Lower/upper estimates",
)
ax.plot(
    yr_df["Year"],
    yr_df["Yearly dose (mSv)"],
    color="navy",
    marker="o",
    label="Yearly dose",
)
ax_daily = ax.twinx()
ax_daily.errorbar(
    daily_stats["Year"],
    daily_stats["average"],
    yerr=[
        daily_stats["average"] - daily_stats["minimum"],
        daily_stats["maximum"] - daily_stats["average"],
    ],
    color="darkorange",
    marker="o",
    capsize=4,
    linestyle="none",
    label="Average daily dose (min-max span)",
)
ax.set_title("Yearly Dose (2010-2021)")
ax.set_xlabel("Year")
ax.set_ylabel("Yearly dose (mSv)", color="navy")
ax.tick_params(axis="y", colors="navy")
ax.spines["left"].set_color("navy")
ax.set_xticks(yr_df["Year"])
ax_daily.set_ylabel("Daily dose (µSv)", color="darkorange")
ax_daily.tick_params(axis="y", colors="darkorange")
ax_daily.spines["right"].set_color("darkorange")
ax.grid(True, alpha=0.3)
handles, labels = ax.get_legend_handles_labels()
daily_handles, daily_labels = ax_daily.get_legend_handles_labels()
ax.legend(handles + daily_handles, labels + daily_labels)
fig.tight_layout()
plt.show()
