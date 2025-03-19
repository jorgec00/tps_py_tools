import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gmplot
import plotly.express as px

class ViperSortieData:
    def __init__(self, filepath: str):
        data = pd.read_csv(filepath)
        # Measured values
        self.time_irig = data["IRIG_TIME"].to_numpy(dtype=str)
        self.angle_of_attack = data["AOA"].to_numpy(dtype=np.float64)
        self.instrument_corrected_altitude = data["BARO_ALT_1553"].to_numpy(dtype=np.float64)
        self.instrument_corrected_airspeed = data["CAL_AS_1553"].to_numpy(dtype=np.float64)
        self.event_marker = data["EVENT_MARKER"].to_numpy(dtype=np.float64)
        self.temp1 = data["FS_TEMP_K_1553"].to_numpy(dtype=np.float64)
        self.true_AoA = data["TRUE_AOA_1553"].to_numpy(dtype=np.float64)
        self.Tic = data["TAT_DEGC"].to_numpy(dtype=np.float64)
        self.angle_of_sideslip = data["AOSS"].to_numpy(dtype=np.float64)
        self.latitude = data["GPS_LATITUDE"].to_numpy(dtype=np.float64)
        self.longitude = data["GPS_LONGITUDE"].to_numpy(dtype=np.float64)

    def extract_events(self, filename: str):
        """
        Extract data for each event marker and save to excel file called filename
        """
        # Get unique event markers
        event_markers = np.unique(self.event_marker)

        # create array to store empty first appearances
        df_first_index = pd.DataFrame()

        # Create a writer
        writer = pd.ExcelWriter(filename)
        # Loop through each event marker
        for event in event_markers:
            # Get indices for this event
            index = np.where(self.event_marker == event)[0][0]
            # Create a dataframe
            df = pd.DataFrame({
                "Event_Marker": [self.event_marker[index]],
                "AoA": [np.average(self.angle_of_attack[index-10:index+10])],
                "Altitude": [np.average(self.instrument_corrected_altitude[index-10:index+10])],
                "Airspeed": [np.average(self.instrument_corrected_airspeed[index-10:index+10])],
                "Temp1": [np.average(self.temp1[index-10:index+10])],
                "True_AoA": [np.average(self.true_AoA[index-10:index+10])],
                "Tic": [np.average(self.Tic[index-10:index+10])],
                "AoSS": [np.average(self.angle_of_sideslip[index-10:index+10])],
                "IRIG_TIME": [self.time_irig[index]]
            })

            # Append to main DF
            df_first_index = pd.concat([df_first_index, df], ignore_index=True)

            # Save to excel
            df_first_index.to_excel(writer)

        # Save the file
        writer.close()
        
        return df_first_index

    def plot_TFB_run(self, event_marker: int):
        """
        Plot the TFB run for a given event marker
        """
        # Get indices for this event
        index = np.where(self.event_marker == event_marker)[0][0]

        # Grab data +/- 3 minutes to plot
        pts = 3*60*20 # 20 Hz data
        lat = self.latitude[index-pts:index+pts]
        lon = self.longitude[index-pts:index+pts]
        Vic = self.instrument_corrected_airspeed[index-pts:index+pts]
        Hic = self.instrument_corrected_altitude[index-pts:index+pts]
        Tic = self.Tic[index-pts:index+pts]
        time = self.time_irig[index-pts:index+pts]

        # Create a dataframe for the main figure
        df_main = pd.DataFrame({
            "time": time,
            "lat": lat,
            "lon": lon,
            "Vic": Vic,
            "Hic": Hic,
            "Tic": Tic
        })

        # Create the main figure
        fig = px.scatter_map(
            df_main,
            lat="lat",
            lon="lon",
            color="Vic",
            color_continuous_scale=px.colors.sequential.Plasma,
            hover_data={
                "Vic": True,
                "Hic": True,
                "Tic": True,
                "time": True,            },
            zoom=11,
            map_style="satellite",
        )

        # Create an event marer
        df_star = pd.DataFrame({
            "lat": self.latitude[index],      # Single point
            "lon": self.longitude[index],     # Single point
            "Vic": self.instrument_corrected_airspeed[index],  # Single point
            "Hic": self.instrument_corrected_altitude[index],  # Single point
            "Tic": self.Tic[index],  # Single point
            "IRIG_TIME": self.time_irig[index],  # Single point
            "event_marker": self.event_marker[index],  # Single point
            "type": ["star"],     # A label for symbol mapping
        })

        # Create a second figure just for the star point
        fig_star = px.scatter_map(
            df_star,
            lat="lat",
            lon="lon",
            hover_data={
                "Vic": True,
                "Hic": True,
                "event_marker": True,
                "Tic": True,
                "IRIG_TIME": True
            },
            color_discrete_sequence=["red"],  # Color for the star poin
            map_style="satellite"
        )

        # Combine
        # Add the star marker trace from 'fig_star' to the main 'fig'
        fig.add_trace(fig_star.data[0])

        # Show everything together
        fig.show()

class TalonSortieData:
    def __init__(self, filepath: str):
        data = pd.read_csv(filepath)
        # Measured values
        self.time_s = data["time_s"].to_numpy(dtype=np.float64)
        self.velocity_north = data["vn_fps"].to_numpy(dtype=np.float64)
        self.velocity_east = data["ve_fps"].to_numpy(dtype=np.float64)
        self.velocity_down = data["vd_fps"].to_numpy(dtype=np.float64)
        self.geometric_height = data["z_ft"].to_numpy(dtype=np.float64)
        self.total_pressure = data["pt_psi"].to_numpy(dtype=np.float64)
        self.static_pressure = data["pa_psi"].to_numpy(dtype=np.float64)
        self.total_temperature = data["tt_k"].to_numpy(dtype=np.float64)
        self.angle_of_attack = data["aoa_rad"].to_numpy(dtype=np.float64)
        self.angle_of_sideslip = data["aos_rad"].to_numpy(dtype=np.float64)
        self.roll = data["phi_rad"].to_numpy(dtype=np.float64)
        self.pitch = data["theta_rad"].to_numpy(dtype=np.float64)
        self.yaw = data["psi_rad"].to_numpy(dtype=np.float64)
        # Derived values
        self.differential_pressure = self.total_pressure - self.static_pressure


def DAS_data_loader():
    # Load DAS data
    print("\nLoading Viper DAS Sortie Data...")
    sortie = ViperSortieData(os.path.join("PF7111", "TFB_20250307_378_DAS_RAW.csv"))

    # Extract event markers and data and save to EXCEL file
    sortie.extract_events(os.path.join("PF7111", "TFB_20250307_378_DAS_EXTRACTED.xlsx"))
    
    # Plot the TFB run for a given event marker
    marker_plot = 2 # Change this to the event marker you want to plot
    print(f"\nPlotting TFB Run for Event Marker {marker_plot}...")
    sortie.plot_TFB_run(marker_plot)
    
if __name__ == "__main__":
    DAS_data_loader()