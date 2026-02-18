#!/usr/bin/env python3
import pandas as pd

class load_gauthier_data:
    def __init__(self, data_path):
        df = pd.read_csv(data_path)

        clin_features_df = df.filter(regex="(clin_.+)|dmax")
        clin_features_df = pd.concat(
            [clin_features_df, df[['patients_id', "pfs_event", "pfs"]]],
            axis=1
        ).drop_duplicates()

        clin_features_df = clin_features_df.drop(
            columns=clin_features_df.filter(regex=r"_\d+").columns).drop(columns=["clin_LDH_categorical"]).drop_duplicates()

        clin_features_df["pfs_2_years"] = ((clin_features_df["pfs_event"] == 1) & (clin_features_df["pfs"] <= 24)).astype(int)

        self.data = clin_features_df
        self.size = clin_features_df.shape[0]
