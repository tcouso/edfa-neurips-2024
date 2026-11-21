from src import paths
import pandas as pd
import json
import yaml


def annotate():
    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    # Metadata
    metadata_df = pd.read_csv(paths.data_raw_dir("metadata.csv"), index_col=["ID", "IDpix"])
    metadata_df["start"] = pd.Timestamp(params["placeholder_date"])
    metadata_df["end"] = pd.Timestamp(params["placeholder_date"])

    # Labels
    labels_df = pd.read_csv(paths.data_external_dir("change_poly_annotations.csv"))

    # Extract polygons IDs
    labels_df['ID'] = labels_df['csv'].str.extract(r'ID_(\d+)')

    # Parse the JSON column to extract 'start' and 'end' dates
    labels_df['label'] = labels_df['label'].apply(lambda x: json.loads(x)[0])
    labels_df['start'] = labels_df['label'].apply(lambda x: pd.to_datetime(x['start']))
    labels_df['end'] = labels_df['label'].apply(lambda x: pd.to_datetime(x['end']))

    labels_df = labels_df[['ID', 'start', 'end']]
    labels_df.index = labels_df["ID"].astype(int)
    labels_df = labels_df.drop("ID", axis=1)

    # Tagging
    for id in labels_df.index:
        metadata_df.loc[id, "start"] = labels_df.loc[id, "start"]
        metadata_df.loc[id, "end"] = labels_df.loc[id, "end"]

    # Saving
    metadata_df.to_csv(paths.data_interim_dir("metadata_annotated.csv"))


