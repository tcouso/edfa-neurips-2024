from src import paths
import pandas as pd
import yaml

def filter():

  with open(paths.config_dir("params.yaml"), "r") as file:
    params = yaml.safe_load(file)

  # Signal
  selected_band = params["selected_band"]
  signal_filename = selected_band + ".csv"
  signal_df = pd.read_csv(
      paths.data_raw_dir(signal_filename), index_col=["ID", "IDpix"]
  )

  # Metadata
  metadata_df = pd.read_csv(paths.data_interim_dir("metadata_annotated.csv"), index_col=["ID", "IDpix"])
  metadata_df["start"] = pd.to_datetime(metadata_df["start"])
  metadata_df["end"] = pd.to_datetime(metadata_df["end"])

  # Remove polygons with start date before training limit date
  changed_polys_df = metadata_df[metadata_df["label"] == 1]
  polys_to_discard = changed_polys_df[changed_polys_df["start"] < pd.Timestamp(params["training_limit_date"])]
  filtered_metadata_df = metadata_df[~metadata_df.index.get_level_values('ID').isin(polys_to_discard.index.get_level_values('ID'))]

  # Apply filter to signal data
  filtered_signal_df = signal_df.loc[filtered_metadata_df.index]

  # Saving
  filtered_signal_df.to_csv(paths.data_interim_dir(selected_band + "_filtered.csv"))
  filtered_metadata_df.to_csv(paths.data_processed_dir("metadata_annotated_filtered.csv"))
