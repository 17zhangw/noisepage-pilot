import pandas as pd
import numpy as np
from pathlib import Path
from behavior import TARGET_COLUMNS


# Any feature that ends with any keyword in BLOCKED_FEATURES is dropped
# from the input schema. These features are intentionally dropped since
# we don't want feature selection/model to try and learn from these.
TRAILING_BLOCKED_FEATURES = [
    "ou_index",
    "data_id",
    "plan_node_id",
    "query_id",
    "db_id",
    "statement_timestamp",
    "pid",
    "left_child_node_id",
    "right_child_node_id",

    "plan_type",
    "cpu_id",
    "relid",
    "indexid",
    "unix_timestamp",
    "relname",
    "relkind",
    "generation",
]


def prepare_index_input_data(df, nodrop, separate_indkey_features):
    # This logic is used to replace all the indkey n_distinct_ data (make it positive).
    distinct_keys = [col for col in df.columns if "indkey_n_distinct_" in col]
    filter_check = lambda x: x >= 0
    other_data = lambda x: (-x) * df["table_reltuples"]
    for distinct_key in distinct_keys:
        # Switch the negative distinct to a positive based on num tuples in table
        df[distinct_key] = df[distinct_key].where(filter_check, other=other_data)
        # make sure there are no longer any negative numbers in distinct...
        assert (df[distinct_key] < 0).sum() == 0

    # TODO(wz2): Here we try to encode summary statistics as much as possible.
    slot = [
        "indkey_attlen_",
        "indkey_atttypmod_",
        "indkey_attvarying_",
        "indkey_avg_width_",

        # TODO(wz2): Null Fraction is dropped since there's no good way of representing this.
        # Models probably also don't need this. Similar reasoning for correlation.
        "indkey_n_distinct_",
        "indkey_null_frac_",
        "indkey_correlation_",
    ]
    blocked = ([col for col in df.columns for prefix in slot if prefix in col])

    # Execute the following sequence only if we are running under nodrop (we want everything)
    # or we don't want separate indkey features.
    if len(blocked) > 0 and (nodrop or not separate_indkey_features):
        varying_keys = [col for col in df.columns if "indkey_attvarying_" in col]
        attlen_keys = [col for col in df.columns if "indkey_attlen_" in col]
        atttypmod_keys = [col for col in df.columns if "indkey_attypmod_" in col]
        avg_width_keys = [col for col in df.columns if "indkey_avg_width_" in col]

        df["indkey_cum_attvarying"] = 0
        df["indkey_cum_attfixed"] = 0
        df["indkey_cum_attlen"] = 0
        df["indkey_cum_avg_width"] = 0
        for col in varying_keys:
            df["indkey_cum_attvarying"] += (df[col].fillna(0))
        for col in avg_width_keys:
            df["indkey_cum_avg_width"] += (df[col].fillna(0))
        for col in attlen_keys:
            # This is a special case where we should fill as (-1).
            df["indkey_cum_attfixed"] += ((df[col].fillna(-1)) != -1).astype(int)
        filter_check = lambda x: x >= 0
        for col in attlen_keys:
            df["indkey_cum_attlen"] += (df[col].fillna(0)).where(filter_check, 0)
        for col in atttypmod_keys:
            df["indkey_cum_attlen"] += (df[col].fillna(0)).where(filter_check, 0)

        if not nodrop:
            df.drop(columns=blocked, inplace=True, errors='raise')
    elif not nodrop and separate_indkey_features:
        # This is the case where we allow dropping and we only care about indkey features.
        # (this is primarily the train case where separate_indkey_features is set).

        # Dropping these is described above.
        refined = [
            "indkey_n_distinct_",
            "indkey_null_frac_",
            "indkey_correlation_",
        ]
        blocked = ([col for col in df.columns for prefix in slot if prefix in col])
        df.drop(columns=blocked, inplace=True, errors='ignore')

    return df


def clean_input_data(df, separate_indkey_features, is_train):
    """
    Function prepares input data for an OU.

    Parameters
    ----------
    df : DataFrame
        Input (train/test) data for an operating unit.
    is_train : bool
        Whether dataframe is used for training. Columns are not dropped for non training dataframes.
    """
    # Remove all features that are blocked.
    blocked = [col for col in df.columns for block in TRAILING_BLOCKED_FEATURES if col.endswith(block)]

    # TODO(wz2): Do we really want to drop all of these?
    # Drop all these additional metadata or useless target columns
    blocked.extend([
        "unix_timestamp",
        "time",
        "txn",
        "payload",
        "comment",

        "indexrelid",
        "indrelid",
        "relname",
        "relnatts", # redundant since just # of key+include in index attributes
        "table_oid",
        "table_reltoastrelid",
        "table_relname",
        "table_relkind",
        "tablename",
        "attname",
        # indnatts - # of key+include in index attributes
        # indnkeyatts - # of key in index attributes
        # table_relnatts -- # of attributes in the table
    ])

    slot = [
        "indkey_attrelid_",
        "indkey_slot_", # index of the pg_attribute
        "indkey_attname_", # name of the attribute
        "indkey_most_common_vals_", # most common vals
        "indkey_most_common_freqs_", # most common freqs
        "indkey_histogram_bounds_", # histogram bounds
        "indkey_tablename_",
        "indkey_attnum_",
    ]
    blocked.extend([] if "indkey" not in df else ["indkey"])
    blocked.extend([col for col in df.columns for prefix in slot if prefix in col])

    # Eliminate all columns that end with OID
    blocked.extend([col for col in df.columns if col.endswith("oid")])

    if df.shape[0] > 0:
        varying_keys = [col for col in df.columns if "indkey_attvarying_" in col]
        for key in varying_keys:
            if df.dtypes[key] == "object":
                df[key] = (df[key] == "t").astype(int)

    # Clean and perform any relevant operations on the input data.
    # If we're training, drop. Otherwise, execute in nodrop.
    df = prepare_index_input_data(df, nodrop=not is_train, separate_indkey_features=separate_indkey_features)

    # This is an OU-specific dataframe operation. Why? Well because we want the num_outer_loops
    # to handle the query-level "nested" behavior.
    if "IndexScan_num_outer_loops" in df.columns:
        df["IndexScan_num_outer_loops"] = np.clip(df.IndexScan_num_outer_loops, 1, None)
        div_cols = TARGET_COLUMNS + ["IndexScan_num_iterator_used", "IndexScan_num_heap_fetches"]
        div_cols = [col for col in div_cols if col in df]
        if len(div_cols) > 0:
            df[div_cols] = df[div_cols].div(df.IndexScan_num_outer_loops, axis=0)
            df.drop(columns=["IndexScan_num_outer_loops"], inplace=True, errors='raise')

    # TODO(wz2): I think we should drop state variables other than below? Why?
    # Because these are transitory and sort of capture the "end" of execution state
    # which will be fundamentally different to when we do inference.
    state_include = ["NumScanKeys", "NumOrderByKeys", "NumRuntimeKeys"]
    state_include = [col for col in df.columns for inc in state_include if inc in col]
    blocked.extend([col for col in df.columns if "ScanState" in col and col not in state_include])

    if is_train:
        # Only drop the columns in train. Evaluate should probably have all the columns for analysis.
        df.drop(blocked, axis=1, inplace=True, errors='ignore')

    df.fillna(0.0, inplace=True)

    # Massage these column datatypes.
    if "indisunique" in df.columns:
        df["indisunique"] = (df.indisunique == "t").astype(int)

    if "indisprimary" in df.columns:
        df["indisprimary"] = (df.indisprimary == "t").astype(int)

    if "indisexclusion" in df.columns:
        df["indisexclusion"] = (df.indisexclusion == "t").astype(int)

    if "indimmediate" in df.columns:
        df["indimmediate"] = (df.indimmediate == "t").astype(int)

    # Convert all bool type columns into an integer column.
    for i, dtype in enumerate(df.dtypes):
        if dtype == "bool":
            df[df.columns[i]] = df[df.columns[i]].astype(int)

    # Sort the DataFrame by column for uniform downstream outputs.
    df.sort_index(axis=1, inplace=True)

    # Shift the target columns to the end.
    df = df[[c for c in df if c not in TARGET_COLUMNS] + [t for t in TARGET_COLUMNS if t in df]]
    return df


class OUDataLoader():
    def _process(self, ou_file, chunk):
        assert chunk.shape[0] > 0
        assert self.loaded == ou_file

        if self.md_plans is not None:
            initial = chunk.shape[0]
            chunk.set_index(keys=["statement_timestamp"], inplace=True)
            chunk.sort_index(axis=0, inplace=True)
            chunk = pd.merge_asof(chunk, self.md_plans, left_index=True, right_index=True, by=["query_id", "generation", "db_id", "pid", "plan_node_id"])
            chunk.reset_index(drop=False, inplace=True)
            chunk.drop(chunk[chunk.indicator.isna()].index, inplace=True)
            chunk.drop(columns=["indicator"], inplace=True)

            # There can only be less, not more.
            assert chunk.shape[0] <= initial

        if self.md_settings is not None:
            initial = chunk.shape[0]
            chunk.set_index(keys=["unix_timestamp"], inplace=True)
            chunk.sort_index(axis=0, inplace=True)
            chunk = pd.merge_asof(chunk, self.md_settings, left_index=True, right_index=True)
            chunk.reset_index(drop=False, inplace=True)
            chunk.drop(chunk[chunk.pg_settings_identifier.isna()].index, inplace=True)
            chunk.drop(columns=["pg_settings_identifier"], inplace=True)
            assert chunk.shape[0] <= initial

        if self.md_tbls is not None:
            initial = chunk.shape[0]
            chunk.set_index(keys=["unix_timestamp"], inplace=True)
            chunk.sort_index(axis=0, inplace=True)
            chunk = pd.merge_asof(chunk, self.md_tbls, left_index=True, right_index=True, left_by=["ModifyTable_target_oid"], right_by=["oid"])
            chunk.reset_index(drop=False, inplace=True)
            chunk.drop(chunk[chunk.oid.isna()].index, inplace=True)
            assert chunk.shape[0] <= initial

        if self.md_idx is not None:
            initial = chunk.shape[0]
            col = ("IndexScan_indexid" if "IndexScan_indexid" in chunk else ("IndexOnlyScan_indexid" if "IndexOnlyScan_indexid" in chunk else "ModifyTableIndexInsert_indexid"))
            assert col in chunk

            if col == "ModifyTableIndexInsert_indexid":
                chunk.drop(columns=["total_cost", "startup_cost"], inplace=True)

            chunk.set_index(keys=["unix_timestamp"], inplace=True)
            chunk.sort_index(axis=0, inplace=True)
            chunk = pd.merge_asof(chunk, self.md_idx, left_index=True, right_index=True, left_by=[col], right_by=["indexrelid"])
            chunk.reset_index(drop=False, inplace=True)
            chunk.drop(chunk[chunk.indexrelid.isna()].index, inplace=True)
            assert chunk.shape[0] <= initial

        chunk.reset_index(drop=True, inplace=True)
        chunk = clean_input_data(chunk, self.separate_indkey_features, self.train).copy()
        chunk["data_identifier"] = chunk.index
        return chunk

    def _load_metadata(self, ou_file):
        if self.loaded == ou_file:
            return

        self.md_plans = None
        self.md_settings = None
        self.md_idx = None
        self.md_tbls = None

        # Here we make assumptions of the files on disk.
        ou = Path(ou_file)
        if (ou.parent / f"{ou.stem}_plan.csv").exists():
            self.md_plans = pd.read_csv(ou.parent / f"{ou.stem}_plan.csv")
            self.md_plans["indicator"] = 1
            self.md_plans.drop(columns=["query_text"], inplace=True, errors='ignore')
            self.md_plans.set_index(keys=["statement_timestamp"], inplace=True)
            self.md_plans.sort_index(axis=0, inplace=True)

        if (ou.parent / f"{ou.stem}_tbls.csv").exists():
            self.md_tbls = pd.read_csv(ou.parent / f"{ou.stem}_tbls.csv")
            self.md_tbls.set_index(keys=["unix_timestamp"], inplace=True)
            self.md_tbls.sort_index(axis=0, inplace=True)

        if (ou.parent / f"{ou.stem}_settings.csv").exists():
            self.md_settings = pd.read_csv(ou.parent / f"{ou.stem}_settings.csv")
            self.md_settings.set_index(keys=["unix_timestamp"], inplace=True)
            self.md_settings.sort_index(axis=0, inplace=True)

        if (ou.parent / f"{ou.stem}_idx.csv").exists():
            self.md_idx = pd.read_csv(ou.parent / f"{ou.stem}_idx.csv")
            self.md_idx.set_index(keys=["unix_timestamp"], inplace=True)
            self.md_idx.sort_index(axis=0, inplace=True)

        self.loaded = ou_file


    def __init__(self, logger, separate_indkey_features, ou_files, chunksize, train):
        super(OUDataLoader, self).__init__()

        self.logger = logger
        self.separate_indkey_features = separate_indkey_features
        self.ou_files = ou_files
        self.chunksize = chunksize
        self.train = train

        self.it = None
        self.finished = False;
        self.loaded = None

        self.md_plans = None
        self.md_settings = None
        self.md_idx = None
        self.md_tbls = None

    def get_next_data(self):
        if self.finished == True:
            return None

        data = None
        while (data is None or data.shape[0] == 0) and len(self.ou_files) > 0:
            current_file = self.ou_files[0]
            self._load_metadata(current_file)

            if self.chunksize is None:
                data = pd.read_csv(current_file)
                self.ou_files = self.ou_files[1:]
            else:
                if self.it is None:
                    self.it = pd.read_csv(current_file, chunksize=self.chunksize)

                try:
                    data = self.it.get_chunk()
                except StopIteration:
                    data = None

                if data is None or data.shape[0] == 0:
                    # We have reached the end of the current file.
                    self.ou_files = self.ou_files[1:]
                    self.it = None
                    data = None

            if data is not None:
                # Try and process the data. We have this here becuase we could end up
                # losing all the data if it is invalid.
                data = self._process(current_file, data)

        if (data is None or data.shape[0] == 0) and len(self.ou_files) == 0:
            self.finished = True
            return None

        return data
