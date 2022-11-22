import glob
import logging
import json
import psycopg
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from sqlalchemy import create_engine
from plumbum import cli
import os

from behavior import OperatingUnit
from behavior import EXECUTION_FEATURES_MAP

from behavior.utils.process_pg_state_csvs import (
    process_time_pg_stats,
    process_time_pg_attribute,
    process_time_pg_class,
    process_time_pg_index,
    process_time_pg_settings,
    postgres_julian_to_unix,

    PG_CLASS_SCHEMA,
    PG_CLASS_INDEX_SCHEMA,
    PG_INDEX_SCHEMA,
    PG_ATTRIBUTE_SCHEMA,
    PG_STATS_SCHEMA,
)
from behavior.utils import read_all_plans

logger = logging.getLogger(__name__)

##################################################################################
# Get query plans
##################################################################################

def prepare_qss_plans(plans_df):
    new_df_tuples = []
    new_plan_features = {}
    for index, row in tqdm(plans_df.iterrows(), total=plans_df.shape[0]):
        feature = json.loads(row.features)
        if len(feature) > 1:
            logger.warn("Skipping decoding of plan: %s", feature)
            continue

        def process_plan(plan):
            plan_features = {}
            for key, value in plan.items():
                if key == "Plans":
                    # For the special key, we recurse into the child.
                    for p in value:
                        process_plan(p)
                    continue

                if isinstance(value, list):
                    # TODO(wz2): For now, we simply featurize a list[] with a numeric length.
                    # This is likely insufficient if the content of the list matters significantly.
                    key = key + "_len"
                    value = len(value)

                plan_features[key] = value

            plan_features = json.dumps(plan_features)

            new_tuple = {
                'query_id': row.query_id,
                'generation': row.generation,
                'db_id': row.db_id,
                'pid': row.pid,
                'statement_timestamp': row.statement_timestamp,
                'plan_node_id': plan["plan_node_id"],

                'left_child_node_id': plan["left_child_node_id"] if "left_child_node_id" in plan else -1,
                'right_child_node_id': plan["right_child_node_id"] if "right_child_node_id" in plan else -1,
                'total_cost': plan["total_cost"],
                'startup_cost': plan["startup_cost"],
                'features': plan_features,
            }
            new_df_tuples.append(new_tuple)

        # For a given query, featurize each node in the plan tree.
        np_dict = feature[0]

        QUERY_SUBSTR_BLOCK = [
            "epoch from NOW()",
            "pg_prewarm",
            "version()",
            "current_schema()",
            "pg_settings",
            # The following are used to block JDBC.
            "pg_catalog.generate_series",
            "n.nspname = 'information_schema'",
            "pg_catalog.pg_namespace",
            # The following are used to block Benchbase.
            "pg_statio_user_indexes",
            "pg_stat_user_indexes",
            "pg_statio_user_tables",
            "pg_stat_user_tables",
            "pg_stat_database_conflicts",
            "pg_stat_database",
            "pg_stat_bgwriter",
            "pg_stat_archiver",
        ]

        skip = False
        query_text = np_dict["query_text"]
        for sub in QUERY_SUBSTR_BLOCK:
            if sub in query_text:
                # Block if the substring can be found.
                # This will then block all corresponding OUs.
                skip = True
                break

        if not skip:
            process_plan(np_dict)

    return pd.DataFrame(new_df_tuples)


def load_plans(data_folder):
    plans = read_all_plans(data_folder)
    plans_df = prepare_qss_plans(plans)
    valid_queries = plans_df[["query_id", "plan_node_id"]].groupby(by=["query_id"]).max() + 1
    plans_df.set_index(keys=["statement_timestamp"], drop=True, append=False, inplace=True)
    plans_df.sort_index(axis=0, inplace=True)
    return plans_df, valid_queries


##################################################################################
# Populate initial table with the raw query stats data
##################################################################################

QSS_STATS_COLUMNS = [
    ("query_id", "bigint"),
    ("generation", "integer"),
    ("db_id", "integer"),
    ("pid", "integer"),
    ("statement_timestamp", "bigint"),
    ("unix_timestamp", "float8"),
    ("plan_node_id", "int"),
    ("left_child_node_id", "int"),
    ("right_child_node_id", "int"),
    ("elapsed_us", "float8"),
    ("counter0", "float8"),
    ("counter1", "float8"),
    ("counter2", "float8"),
    ("counter3", "float8"),
    ("counter4", "float8"),
    ("counter5", "float8"),
    ("counter6", "float8"),
    ("counter7", "float8"),
    ("counter8", "float8"),
    ("counter9", "float8"),
    ("blk_hit", "integer"),
    ("blk_miss", "integer"),
    ("blk_dirty", "integer"),
    ("blk_write", "integer"),
    ("startup_cost", "float8"),
    ("total_cost", "float8"),
    ("payload", "bigint"),
    ("txn", "bigint"),
    ("comment", "text"),
]

DIFFERENCE_COLUMNS = [
    "elapsed_us",
    "blk_hit",
    "blk_miss",
    "blk_dirty",
    "blk_write",
    "startup_cost",
    "total_cost",
]


def construct_diff_sql(work_prefix, raw, all_columns, diff_columns, constrain_child=False):
    diff_sql = "SELECT "
    for i in range(len(all_columns)):
        k = all_columns[i][0]
        if k not in diff_columns:
            diff_sql += f"r1.{k}"
        else:
            diff_sql += f"greatest(r1.{k} - coalesce(r2.{k}, 0) - coalesce(r3.{k}, 0), 0) as {k}"

        if i != len(all_columns) - 1:
            diff_sql += ", "
    diff_sql += f" FROM {work_prefix}_{raw} r1 "
    diff_sql += f"LEFT JOIN {work_prefix}_{raw} r2 "
    diff_sql += """
        ON r1.query_id = r2.query_id AND
           r1.db_id = r2.db_id AND
           r1.statement_timestamp = r2.statement_timestamp AND
           r1.pid = r2.pid AND
           r1.left_child_node_id = r2.plan_node_id AND
           r1.plan_node_id >= 0
    """
    if constrain_child:
        diff_sql += " AND r2.plan_node_id >= 0\n"

    diff_sql += f"LEFT JOIN {work_prefix}_{raw} r3 "
    diff_sql += """
        ON r1.query_id = r3.query_id AND
           r1.db_id = r3.db_id AND
           r1.statement_timestamp = r3.statement_timestamp AND
           r1.pid = r3.pid AND
           r1.right_child_node_id = r3.plan_node_id AND
           r1.plan_node_id >= 0
    """
    if constrain_child:
        diff_sql += " AND r3.plan_node_id >= 0\n"
    return diff_sql


def load_initial_data(connection, data_folder, work_prefix, plans_df):
    with open(f"/tmp/{work_prefix}_stats.csv", "w") as f:
        write_header = True
        stats_files = [f for f in glob.glob(f"{data_folder}/stats.*/pg_qss_stats_*.csv")]
        for stats_file in tqdm(stats_files, leave=False):
            for chunk in tqdm(pd.read_csv(stats_file, chunksize=8192*1000), leave=False):
                chunk = chunk[(chunk.plan_node_id != -1) & (chunk.query_id != 0) & (chunk.statement_timestamp != 0)]
                chunk = chunk[chunk.query_id.isin(plans_df.query_id)]
                if chunk.shape[0] == 0:
                    continue

                if chunk["query_id"].dtype != "int64":
                    # Convert to int64 if needed.
                    chunk = chunk.copy()
                    chunk["query_id"] = chunk.query_id.astype(np.int64)

                # These have no matching plan, so we actually need to drop them...
                # Any costs are pretty much meaningless.
                noplans = chunk[chunk.plan_node_id < 0].copy()
                # Wipe the child plans.
                noplans["left_child_node_id"] = -1
                noplans["right_child_node_id"] = -1

                # These should have valid plans attached to them
                plans = chunk[chunk.plan_node_id >= 0].copy()
                plans.set_index(keys=["statement_timestamp"], drop=True, append=False, inplace=True)
                plans.sort_index(axis=0, inplace=True)

                chunk = pd.merge_asof(plans, plans_df, left_index=True, right_index=True, by=["query_id", "generation", "db_id", "pid", "plan_node_id"], allow_exact_matches=True)
                chunk.reset_index(drop=False, inplace=True)
                # Assert that we actually don't have any missing plans.
                # chunk.drop(chunk[chunk.total_cost.isna()].index, inplace=True)
                assert chunk.total_cost.isna().sum() == 0

                # Recombine.
                chunk = pd.concat([chunk, noplans], ignore_index=True)
                chunk["unix_timestamp"] = postgres_julian_to_unix(chunk.statement_timestamp)

                chunk = chunk[[t[0] for t in QSS_STATS_COLUMNS]]
                if chunk.shape[0] == 0:
                    continue

                chunk["left_child_node_id"] = chunk.left_child_node_id.astype(int)
                chunk["right_child_node_id"] = chunk.right_child_node_id.astype(int)
                chunk.to_csv(f, header=write_header, index=False)
                write_header = False

    with connection.transaction():
        create_stats_sql = f"CREATE UNLOGGED TABLE {work_prefix}_raw (" + ",".join([f"{tup[0]} {tup[1]}" for tup in QSS_STATS_COLUMNS]) + ")"
        connection.execute(create_stats_sql)
        connection.execute(f"COPY {work_prefix}_raw FROM '/tmp/{work_prefix}_stats.csv' WITH (FORMAT csv, HEADER true)")
        connection.execute(f"CREATE INDEX {work_prefix}_raw_0 ON {work_prefix}_raw (query_id, db_id, pid, statement_timestamp)")
        connection.execute(f"CREATE UNIQUE INDEX {work_prefix}_raw_1 ON {work_prefix}_raw (query_id, db_id, pid, statement_timestamp, plan_node_id) WHERE plan_node_id >= 0")
    os.remove(f"/tmp/{work_prefix}_stats.csv")


def load_filter(connection, work_prefix, valid_queries):
    queries = valid_queries.reset_index(drop=False)
    with connection.transaction():
        create_stats_sql = f"CREATE UNLOGGED TABLE {work_prefix}_raw_filter (" + ",".join([f"{tup[0]} {tup[1]}" for tup in QSS_STATS_COLUMNS]) + ")"
        connection.execute(create_stats_sql)
        connection.execute(f"INSERT INTO {work_prefix}_raw_filter SELECT * FROM {work_prefix}_raw WHERE plan_node_id < 0")
        for q in queries.itertuples():
            all_cols = ",".join([tup[0] for tup in QSS_STATS_COLUMNS])
            ins_sel = f"INSERT INTO {work_prefix}_raw_filter (" + all_cols + ") "
            ins_sel += "SELECT " + all_cols + " FROM "

            ins_sel += "( SELECT " + all_cols + ", COUNT(1) OVER (PARTITION BY query_id, db_id, pid, statement_timestamp) as cnt "
            ins_sel += f" FROM {work_prefix}_raw WHERE plan_node_id >= 0 AND query_id = {q.query_id}) as t "
            ins_sel += f"WHERE t.cnt = {q.plan_node_id}"
            connection.execute(ins_sel)

        # Indexes are cheap so just make them.
        connection.execute(f"CREATE INDEX {work_prefix}_raw_filter_0 ON {work_prefix}_raw_filter (query_id, db_id, pid, statement_timestamp)")
        connection.execute(f"CREATE UNIQUE INDEX {work_prefix}_raw_filter_1 ON {work_prefix}_raw_filter (query_id, db_id, pid, statement_timestamp, plan_node_id) WHERE plan_node_id >= 0")
        connection.execute(f"CREATE INDEX {work_prefix}_raw_filter_2 ON {work_prefix}_raw_filter (query_id, db_id, pid, statement_timestamp, plan_node_id)")
        connection.execute(f"CLUSTER {work_prefix}_raw_filter USING {work_prefix}_raw_filter_2")

##################################################################################
# Create the differencing data
##################################################################################

def diff_data(connection, work_prefix):
    with connection.transaction():
        diff_sql = f"CREATE UNLOGGED TABLE {work_prefix}_diff (" + ",".join([f"{tup[0]} {tup[1]}" for tup in QSS_STATS_COLUMNS]) + ") PARTITION BY LIST (comment)"
        connection.execute(diff_sql)

        for ou in OperatingUnit:
            connection.execute(f"CREATE UNLOGGED TABLE {work_prefix}_diff_{ou.name} PARTITION OF {work_prefix}_diff FOR VALUES IN ('{ou.name}') WITH (autovacuum_enabled = OFF)")
        connection.execute(f"CREATE UNLOGGED TABLE {work_prefix}_diff_default PARTITION OF {work_prefix}_diff DEFAULT WITH (autovacuum_enabled = OFF)")

        diff_sql = f"INSERT INTO {work_prefix}_diff " + construct_diff_sql(work_prefix, "raw_filter", QSS_STATS_COLUMNS, DIFFERENCE_COLUMNS, constrain_child=False)
        connection.execute(diff_sql)

##################################################################################
# Load plans and features
##################################################################################

CREATE_PLAN_COLUMNS = [
    "query_id",
    "generation",
    "db_id",
    "pid",
    "statement_timestamp",
    "plan_node_id",
    "features"
]

def load_metadata(connection, engine, data_dir, work_prefix, plans_df):
    plans_df = plans_df.reset_index(drop=False)
    with engine.begin() as alchemy:
        pg_attribute = process_time_pg_attribute(pd.read_csv(f"{data_dir}/pg_attribute.csv", low_memory=False))
        pg_stats = process_time_pg_stats(pd.read_csv(f"{data_dir}/pg_stats.csv"))
        pg_index = process_time_pg_index(pd.read_csv(f"{data_dir}/pg_index.csv"))
        time_tables, time_cls_indexes = process_time_pg_class(pd.read_csv(f"{data_dir}/pg_class.csv"))
        time_pg_settings = process_time_pg_settings(pd.read_csv(f"{data_dir}/pg_settings.csv"))

        plans_df[CREATE_PLAN_COLUMNS].to_sql(f"{work_prefix}_md_plans", alchemy, index=False)
        pg_attribute.to_sql(f"{work_prefix}_md_pg_attribute", alchemy, index=False)
        pg_stats.to_sql(f"{work_prefix}_md_pg_stats", alchemy, index=False)
        pg_index.to_sql(f"{work_prefix}_md_pg_index", alchemy, index=False)
        time_tables.to_sql(f"{work_prefix}_md_pg_class_tables", alchemy, index=False)
        time_cls_indexes.to_sql(f"{work_prefix}_md_pg_class_indexes", alchemy, index=False)
        time_pg_settings.to_sql(f"{work_prefix}_md_pg_settings", alchemy, index=False)

    with connection.transaction():
        connection.execute(f"CREATE INDEX {work_prefix}_md_plans_0 ON {work_prefix}_md_plans(query_id, generation, db_id, pid, plan_node_id, statement_timestamp)")
        connection.execute(f"ALTER TABLE {work_prefix}_md_plans ALTER COLUMN features TYPE json USING features::json")

        connection.execute(f"CREATE UNIQUE INDEX {work_prefix}_md_pg_class_indexes_0 ON {work_prefix}_md_pg_class_indexes (oid, unix_timestamp)")
        connection.execute(f"CREATE UNIQUE INDEX {work_prefix}_md_pg_index_0 ON {work_prefix}_md_pg_index (indexrelid, unix_timestamp)")
        connection.execute(f"ALTER TABLE {work_prefix}_md_pg_index ALTER COLUMN indkey TYPE int2vector USING indkey::int2vector")

        connection.execute(f"CREATE UNIQUE INDEX {work_prefix}_md_pg_attribute_0 ON {work_prefix}_md_pg_attribute (attrelid, attnum, unix_timestamp)")
        connection.execute(f"CREATE UNIQUE INDEX {work_prefix}_md_pg_stats_0 ON {work_prefix}_md_pg_stats (tablename, attname, unix_timestamp)")
        connection.execute(f"CREATE UNIQUE INDEX {work_prefix}_md_pg_settings_0 ON {work_prefix}_md_pg_settings (unix_timestamp)")
        connection.execute(f"CREATE UNIQUE INDEX {work_prefix}_md_pg_class_tables_0 ON {work_prefix}_md_pg_class_tables (oid, unix_timestamp)")

        connection.execute(f"CLUSTER {work_prefix}_md_pg_class_indexes USING {work_prefix}_md_pg_class_indexes_0")
        connection.execute(f"CLUSTER {work_prefix}_md_pg_index USING {work_prefix}_md_pg_index_0")
        connection.execute(f"CLUSTER {work_prefix}_md_pg_attribute USING {work_prefix}_md_pg_attribute_0")
        connection.execute(f"CLUSTER {work_prefix}_md_pg_stats USING {work_prefix}_md_pg_stats_0")
        connection.execute(f"CLUSTER {work_prefix}_md_pg_settings USING {work_prefix}_md_pg_settings_0")

    with connection.transaction():
        with connection.execute(f"SELECT max(array_length(idx.indkey, 1)) FROM {work_prefix}_md_pg_index idx") as cur:
            max_indkey_size = cur.fetchall()[0][0]

        create_view = f"CREATE MATERIALIZED VIEW {work_prefix}_md_idx_view AS SELECT cls_idx.unix_timestamp, "
        create_view += ",\n".join([f"cls_idx.{c}" for c in PG_CLASS_INDEX_SCHEMA]) + "," 
        create_view += ",\n".join(PG_INDEX_SCHEMA) + "," 
        create_view += ",\n".join([f"tbls.{c} as table_{c}" for c in PG_CLASS_SCHEMA]) + ","
        create_view += ",\n".join([f"NULLIF(split_part(atts.{col}_agg, ',', {i+1}), '') as indkey_{col}_{i}" for col in PG_ATTRIBUTE_SCHEMA for i in range(max_indkey_size)]) + ","
        create_view += ",\n".join([f"NULLIF(split_part(atts.{col}_agg, ',', {i+1}), '') as indkey_{col}_{i}" for col in PG_STATS_SCHEMA for i in range(max_indkey_size) if col not in PG_ATTRIBUTE_SCHEMA]) + ","
        create_view += ",\n".join([f"(NULLIF(split_part(atts.attlen_agg, ',', {i+1}), '')::int = -1)::bool as indkey_attvarying_{i}" for i in range(max_indkey_size)])

        create_view += f" FROM {work_prefix}_md_pg_class_indexes cls_idx, "
        create_view += f"LATERAL (SELECT idx.* FROM {work_prefix}_md_pg_index idx "
        create_view += """
                 WHERE cls_idx.oid = idx.indexrelid
                   AND cls_idx.unix_timestamp >= idx.unix_timestamp
              ORDER BY idx.unix_timestamp DESC LIMIT 1
            ) idx,
            LATERAL (
        """
        create_view += f"SELECT tbls.* FROM {work_prefix}_md_pg_class_tables tbls "
        create_view += """
                 WHERE idx.indrelid = tbls.oid
                   AND cls_idx.unix_timestamp >= tbls.unix_timestamp
              ORDER BY tbls.unix_timestamp DESC LIMIT 1
            ) tbls,
        """

        create_view += "LATERAL (SELECT "
        create_view += ",\n".join([f"string_agg(tt.{col}::text, ',') as {col}_agg" for col in PG_ATTRIBUTE_SCHEMA]) + ","
        create_view += ",\n".join([f"string_agg(tt.{col}::text, ',') as {col}_agg" for col in PG_STATS_SCHEMA if col not in PG_ATTRIBUTE_SCHEMA])
        create_view += " FROM (SELECT atts.*, "
        create_view += ",".join([f"stts.{col}" for col in PG_STATS_SCHEMA if col not in PG_ATTRIBUTE_SCHEMA])
        create_view += f" FROM {work_prefix}_md_pg_attribute atts, LATERAL (SELECT stats.* FROM {work_prefix}_md_pg_stats stats "
        create_view += """
                         WHERE stats.tablename = tbls.relname
                           AND stats.attname = atts.attname
                           AND cls_idx.unix_timestamp >= stats.unix_timestamp
                        ORDER BY stats.unix_timestamp DESC
                        LIMIT 1
                    ) stts
                 WHERE atts.attrelid = idx.indrelid
                   AND atts.attnum = ANY(idx.indkey)
                   AND cls_idx.unix_timestamp >= atts.unix_timestamp
              ORDER BY atts.unix_timestamp DESC, array_position(idx.indkey, atts.attnum)
		      LIMIT array_length(idx.indkey, 1)
		    ) tt
        ) atts;
        """
        connection.execute(create_view)
        connection.execute(f"CREATE INDEX {work_prefix}_md_idx_view_0 on {work_prefix}_md_idx_view (indexrelid, unix_timestamp DESC)")


##################################################################################
# Extract
##################################################################################

def extract_ous(connection, work_prefix, plans_df, output_dir):
    with connection.transaction():
        for ou in OperatingUnit:
            logger.info(f"Processing {ou.name} from difference database.")
            queries = []

            derived_map = {}
            for k, v in EXECUTION_FEATURES_MAP.items():
                if k.startswith(ou.name):
                    derived_map[v] = k

            query = "SELECT " + ",".join([tup[0] for tup in QSS_STATS_COLUMNS if not tup[0].startswith("counter")])
            # Get all the derived execution features from renaming counters.
            if len(derived_map) > 0:
                query += "," + ",".join([f"{k} as \"{v}\"" for k, v in derived_map.items()])
            query += f" FROM {work_prefix}_diff WHERE comment = '{ou.name}'"
            queries.append((query, f"{ou.name}.csv"))

            all_features = set()
            for tup in plans_df.itertuples():
                features = json.loads(tup.features)
                if features["node_type"] == ou.name:
                    # Get all the features that are not "these" features.
                    all_features.update([f for f in features if f not in [
                        "startup_cost",
                        "total_cost",
                        "node_type",
                        "plan_node_id",
                        "left_child_node_id", 
                        "right_child_node_id",
                    ]])

            if len(all_features) > 0:
                query = "SELECT " + ",".join([f for f in CREATE_PLAN_COLUMNS if f != "features"])
                query += "," + ",".join([f"(plan.features->>'{col}') as \"{col}\" " for col in all_features])
                # We require having this node_type field be there.
                query += f" FROM {work_prefix}_md_plans plan WHERE plan.features->>'node_type' = '{ou.name}'"
                queries.append((query, f"{ou.name}_plan.csv"))

            if ou in [OperatingUnit.IndexScan, OperatingUnit.IndexOnlyScan, OperatingUnit.ModifyTableIndexInsert]:
                query = f"SELECT * FROM {work_prefix}_md_pg_settings"
                queries.append((query, f"{ou.name}_settings.csv"))

                query = f"SELECT * FROM {work_prefix}_md_idx_view"
                queries.append((query, f"{ou.name}_idx.csv"))
            elif ou == OperatingUnit.ModifyTableInsert or ou == OperatingUnit.ModifyTableUpdate:
                query = f"SELECT * FROM {work_prefix}_md_pg_class_tables "
                queries.append((query, f"{ou.name}_tbls.csv"))

            ou_path = output_dir / ou.name
            ou_path.mkdir(parents=True, exist_ok=True)

            logger.info("%s", datetime.now())
            for (q, loc) in queries:
                q = f"COPY ({q}) TO '{ou_path}/{loc}' DELIMITER ',' CSV HEADER"
                logger.info("%s -> %s", q, loc)
                connection.execute(q)
            logger.info("%s", datetime.now())

##################################################################################
# Purge DB
##################################################################################

def purge(connection, work_prefix):
    with connection.transaction():
        connection.execute(f"DROP TABLE IF EXISTS {work_prefix}_raw")
        connection.execute(f"DROP TABLE IF EXISTS {work_prefix}_raw_filter")
        connection.execute(f"DROP TABLE IF EXISTS {work_prefix}_diff")

        connection.execute(f"DROP MATERIALIZED VIEW IF EXISTS {work_prefix}_md_idx_view")
        connection.execute(f"DROP TABLE IF EXISTS {work_prefix}_md_plans")
        connection.execute(f"DROP TABLE IF EXISTS {work_prefix}_md_pg_attribute")
        connection.execute(f"DROP TABLE IF EXISTS {work_prefix}_md_pg_stats")
        connection.execute(f"DROP TABLE IF EXISTS {work_prefix}_md_pg_index")
        connection.execute(f"DROP TABLE IF EXISTS {work_prefix}_md_pg_class_tables")
        connection.execute(f"DROP TABLE IF EXISTS {work_prefix}_md_pg_class_indexes")
        connection.execute(f"DROP TABLE IF EXISTS {work_prefix}_md_pg_settings")


def main(dir_data, experiment, dir_output, work_prefix, host, port, db_name, user, preserve):
    logger.info("Extracting OU features for experiment: %s", experiment)
    conn_str = f"host={host} port={port} dbname={db_name} user={user}"
    engine = create_engine(f"postgresql://{user}@{host}:{port}/{db_name}")
    with psycopg.connect(conn_str, autocommit=False, prepare_threshold=None) as connection:
        experiment_root: Path = dir_data / experiment
        bench_names: list[str] = [d.name for d in experiment_root.iterdir() if d.is_dir()]
        for bench_name in bench_names:
            data_folder = experiment_root / bench_name
            ou_stats = [s for s in glob.glob(f"{data_folder}/stats.*/pg_qss_stats_*.csv")]
            if len(ou_stats) == 0:
                continue

            logger.info("Benchmark: %s", bench_name)

            # Purge prior work.
            purge(connection, work_prefix)

            # Construct and load all the plans.
            plans_df, valid_queries = load_plans(data_folder)

            # Time ~5 minutes for 60 million rows.
            logger.info("Populating with raw data %s", datetime.now())
            load_initial_data(connection, data_folder, work_prefix, plans_df)
            logger.info("Finished populating with raw data %s", datetime.now())

            # Time ~2 minutes.
            logger.info("Populating filtered raw data %s", datetime.now())
            load_filter(connection, work_prefix, valid_queries)
            logger.info("Finished populating filtered raw data %s", datetime.now())

            # Time ~3 minutes.
            logger.info("Differencing data %s", datetime.now())
            diff_data(connection, work_prefix)
            logger.info("Finished differencing data %s", datetime.now())

            # Time a short while.
            logger.info("Loading metadata %s", datetime.now())
            load_metadata(connection, engine, data_folder, work_prefix, plans_df)
            logger.info("Finished loading metadata %s", datetime.now())

            # Get those OUs!
            output_data_dir: Path = dir_output / experiment / bench_name
            output_data_dir.mkdir(parents=True, exist_ok=True)
            extract_ous(connection, work_prefix, plans_df, output_data_dir)

            if not preserve:
                # Purge if we don't want to keep it.
                purge(connection, work_prefix)


class ExtractOUsCLI(cli.Application):
    dir_data = cli.SwitchAttr("--dir-data", Path, mandatory=True, help="Root of path to look for experiments matching the glob.")
    dir_output = cli.SwitchAttr("--dir-output", Path, mandatory=True, help="Path to output the extracted OUs to.")
    glob_pattern = cli.SwitchAttr("--glob-pattern", mandatory=False, help="Glob pattern to use for selecting valid experiments.")
    work_prefix = cli.SwitchAttr("--work-prefix", str, mandatory=True, help="Prefix to use for creating and operating tables.")

    host = cli.SwitchAttr("--host", str, mandatory=True, default="localhost", help="Host of the database instance to use.")
    port = cli.SwitchAttr("--port", str, mandatory=True, default='5432', help="Port of the database instance to connect to.")
    db_name = cli.SwitchAttr("--db-name", str, mandatory=True, help="Name of the databse to use.")
    user = cli.SwitchAttr("--user", str, mandatory=True, help="Username to use to connect.")
    preserve = cli.Flag("--preserve", default=False, help="Whether to preserve database or not in single setups.")

    def main(self):
        # By default, difference all the valid experiments.
        pattern = "*" if self.glob_pattern is None else self.glob_pattern
        experiments = sorted(path.name for path in self.dir_data.glob(pattern))
        assert len(experiments) > 0, "No training data found?"

        for experiment in experiments:
            main(self.dir_data,
                 experiment,
                 self.dir_output,
                 self.work_prefix,
                 self.host,
                 self.port,
                 self.db_name,
                 self.user,
                 self.preserve
            )


if __name__ == "__main__":
    ExtractOUsCLI.run()
