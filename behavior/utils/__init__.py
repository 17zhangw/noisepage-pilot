import numpy as np
import pandas as pd
import glob

def read_all_plans(input_dir):
    plans = glob.glob(f"{input_dir}/stats.*/pg_qss_plans_*.csv")
    def read_csv(file):
        data = pd.read_csv(file)
        data["query_id"] = data.query_id.astype(np.int64)
        return data
    return pd.concat(map(read_csv, plans), ignore_index=True)
