import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class ViperSortieData:
    def __init__(self, filepath: str):
        data = pd.read_csv(filepath)
        # Measured values
        #self.time_irig = data["IRIG_TIME"].to_numpy(dtype=np.float64)
        self.angle_of_attack = data["AOA"].to_numpy(dtype=np.float64)
        self.instrument_corrected_altitude = data["BARO_ALT_1553"].to_numpy(dtype=np.float64)
        self.instrument_corrected_airspeed = data["CAL_AS_1553"].to_numpy(dtype=np.float64)
        self.event_marker = data["EVENT_MARKER"].to_numpy(dtype=np.float64)
        self.temp1 = data["FS_TEMP_K_1553"].to_numpy(dtype=np.float64)
        self.true_AoA = data["TRUE_AOA_1553"].to_numpy(dtype=np.float64)
        self.temp2 = data["TAT_DEGC"].to_numpy(dtype=np.float64)
        self.angle_of_sideslip = data["AOSS"].to_numpy(dtype=np.float64)

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
                "Temp2": [np.average(self.temp2[index-10:index+10])],
                "AoSS": [np.average(self.angle_of_sideslip[index-10:index+10])]
            })

            # Append to main DF
            df_first_index = pd.concat([df_first_index, df], ignore_index=True)

            # Save to excel
            df_first_index.to_excel(writer)

        # Save the file
        writer.close()


def data_loader():
    # Load DAS data
    print("\nLoading Viper DAS Sortie Data...")
    sortie = ViperSortieData(os.path.join("PF7111", "TFB_20250307_378_DAS_RAW.csv"))

    # Extract event markers and data and save to EXCEL file
    sortie.extract_events(os.path.join("PF7111", "TFB_20250307_378_DAS_EXTRACTED.xlsx"))
    
    # Print data summary
    print("\nData Summary:")
    print(pd.Series(sortie.event_marker).describe())
    plt.plot(sortie.event_marker)
    plt.show()
    
if __name__ == "__main__":
    data_loader()