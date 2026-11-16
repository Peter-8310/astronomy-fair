import xarray as xr
import pandas as pd
import os

EPOCH_MY24_LS0 = pd.Timestamp("1998-07-14 21:21:36", tz="UTC")
SECONDS_PER_SOL = 88775.244147

def sol_to_year(sols: pd.Series) -> tuple[pd.Series, pd.Series]:
    utc_time = EPOCH_MY24_LS0 + pd.to_timedelta(
        sols * SECONDS_PER_SOL, unit="s"
    )

    year = utc_time.dt.year
    year_start = pd.to_datetime(year.astype(str) + "-01-01", utc=True)
    next_year_start = pd.to_datetime(
        (year + 1).astype(str) + "-01-01", utc=True
    )

    year_fraction = (utc_time - year_start) / (next_year_start - year_start)
    earth_fractional_year = year + year_fraction

    return utc_time, earth_fractional_year

def process_nc(filepath:str) -> pd.DataFrame:
    ds = xr.open_dataset(filepath)

    temp_data = ds["tsurf"].sel(lat=47.2, lon=184.3, method="nearest")
    pressure_data = ds["ps"].sel(lat=47.2, lon=184.3, method="nearest")

    df_tsurf = temp_data.to_dataframe().reset_index()
    df_ps = pressure_data.to_dataframe().reset_index()

    df_tsurf -= 273.15
    utc, earth_time = sol_to_year(df_tsurf['time'])

    surf_df = pd.concat([
            df_tsurf['time'], 
            earth_time, 
            utc.dt.year, 
            utc.dt.month, 
            utc.dt.day, 
            utc.dt.hour,
            utc.dt.minute,
            utc.dt.second, 
            df_tsurf['tsurf'], 
            df_ps['ps']
        ], axis=1)

    cols = list(surf_df.columns)

    cols[0] = "Martian time (sol)"
    cols[1] = "Fractional year"
    cols[2] = "Year"
    cols[3] = "Month"
    cols[4] = "Day"
    cols[5] = "Hour"
    cols[6] = "Minute"
    cols[7] = "Second"
    cols[8] = "Surface Temperature (°C)"
    cols[9] = "Pressure (Pa)"

    surf_df.columns = cols

    return surf_df

def main():
    mars_df = pd.DataFrame()
    size = sum(i not in ['OpenMARS-MY28-35-reference-manual.pdf', 'sample_data_plot.py', 'MCS_ret_coverage.nc', 'my35_ls95_temp.png'] for i in os.listdir('Mars_surface_dataset'))

    i = 0
    for  filename in os.listdir('Mars_surface_dataset'):
        if filename not in ['OpenMARS-MY28-35-reference-manual.pdf', 'sample_data_plot.py', 'MCS_ret_coverage.nc', 'my35_ls95_temp.png']:
            i+=1
            print(f"Processing file: {filename} ({i}/{size})")
            mars_df = pd.concat([mars_df, process_nc(f"Mars_surface_dataset/{filename}")], ignore_index=True)
    
    print("Mars Dataset processed")    
    print(mars_df)
    mars_df.to_csv("mars_surface_data.csv", index=False)
    print("Saved to csv file")


if __name__ == "__main__":
    main()