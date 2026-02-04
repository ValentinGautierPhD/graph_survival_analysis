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
            columns=["clin_Ann_Arbor_stage","clin_ECOG_scale","clin_aaIPI",
                     "clin_aaIPI_1.1","clin_aaIPI_2.1","clin_aaIPI_3.1"]).drop_duplicates()

        clin_features_df["pfs_2_years"] = ((clin_features_df["pfs_event"] == 1) & (clin_features_df["pfs"] <= 24)).astype(int)

        self.data = clin_features_df
        self.size = clin_features_df.shape[0]
