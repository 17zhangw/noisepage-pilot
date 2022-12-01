import copy
import json
from tqdm import tqdm
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from psycopg.rows import dict_row
from pandas.api import types as pd_types

from behavior import OperatingUnit
from behavior.model_workload.utils import OpType


def __execute_dist_query(logger, cursor, work_prefix, tbl, att_keys, query_orders, buckets):
    # NOTE that we assume the following of query_orders.
    #
    # window=0 contains all queries [query_orders[0], query_orders[1])
    # window=1 contains all queries [query_orders[1], query_orders[2])
    # ...
    #
    # By this definition, query_orders are window start points!
    query_orders = [str(q) for q in query_orders]

    output_tuples = []
    def create_histograms(window, optype, index_clause, maintenance_body, normalizer):
        atts = maintenance_body.keys()
        for att in atts:
            subdict = maintenance_body[att]
            if len(subdict) == 0:
                # Window did not do anything to this key.
                continue

            keys = [k for k in subdict.keys()]
            values = [subdict[k] for k in keys]

            min_k = normalizer[f"min_{att}"]
            max_k = normalizer[f"max_{att}"]
            keys = [1.0 if min_k == max_k else (k - min_k) / (max_k - min_k) for k in keys]

            # Guarantee that we are already in the 0.0 and 1.0 range.
            assert np.max(keys) <= 1.0
            assert np.min(keys) >= 0.0

            bins, _ = np.histogram(keys, bins=buckets, range=(0.0, 1.0), weights=values)
            bins = bins.astype(np.float)
            assert np.sum(bins) > 0
            bins /= np.sum(bins)

            output_tuples.append([window, optype, index_clause, att, bins.tolist()])

    # Generate the query template.
    query_template = """
        SELECT {extract_columns}, b.bucket, count(1) as count
        FROM (
            SELECT *, width_bucket({column}, ARRAY[{query_orders}]) as bucket
            FROM {work_prefix}_{tbl} {filter}) b
        GROUP BY GROUPING SETS ({groups});
    """

    # First get all the relevant data.
    def acquire_base_data():
        # The trick we observe here is that "data" that started in the database has an unset insert_version.
        # This unset insert_version means that it'll get assigned to bucket = 0 since they'll be less than query_orders[0].
        #
        # Furthermore, "data" is concerned with the data when the window executes so we don't (1-shift) the query_orders.
        # "data": window-0 is technically data that exists before query_orders[0]
        #       : window-1 is data that exists before query_orders[1].
        #
        # Compute the inserts associated with each window.
        extract_columns = [f"b.{k}" for k in att_keys]
        groups = [f"(b.bucket, b.{k})" for k in att_keys]
        insert_frame = query_template.format(
            extract_columns=",".join(extract_columns),
            column="insert_version",
            query_orders=",".join(query_orders),
            work_prefix=work_prefix,
            tbl=tbl,
            filter="",
            groups=",".join(groups)
        )
        insert_rows = [r for r in cursor.execute(insert_frame)]

        # Compute the deletes associated with each window.
        delete_frame = query_template.format(
            extract_columns=",".join(extract_columns),
            column="delete_version",
            query_orders=",".join(query_orders),
            work_prefix=work_prefix,
            tbl=tbl,
            filter="WHERE delete_version > 0",
            groups=",".join(groups)
        )
        insert_rows.extend([r for r in cursor.execute(delete_frame)])

        # Create the joint frame.
        columns = att_keys + ["bucket", "freq_count"]
        data_frame = pd.DataFrame(insert_rows, columns=columns)
        data_frame["optype"] = "data"
        data_frame["index_used"] = None

        # When computing the keyspace touched, we need to 1-shift the query orders.
        # This is because we want all queries between [query_orders[0], query_orders[1]) to be mapped to 0.
        # This is equivalent to dropping query_orders[0]
        extract_columns = [f"b.{k}" for k in att_keys] + ["b.optype", "b.index_used"]
        columns = att_keys + ["optype", "index_used", "bucket", "freq_count"]
        groups = [f"(b.bucket, b.optype, b.index_used, b.{k})" for k in att_keys]
        insert_frame = query_template.format(
            extract_columns=",".join(extract_columns),
            column="query_order",
            query_orders=",".join(query_orders[1:]),
            work_prefix=work_prefix,
            tbl=tbl + "_hits",
            filter="",
            groups=",".join(groups)
        )

        # Now we have everything in one huge frame.
        data_frame = pd.concat([data_frame, pd.DataFrame([r for r in cursor.execute(insert_frame)], columns=columns)], ignore_index=True)

        # Tracks the actual data on disk evolution across windows.
        content_body = {k: {} for k in att_keys}
        content_range_body = {k: {} for k in att_keys}
        window_normalizer = {}
        frame = data_frame.sort_values(by=["bucket"], ignore_index=True)
        def add_normalizer(bucket, content_body, touch_body):
            window_normalizer[bucket] = {}
            for k in content_body:
                min_k = content_range_body[f"min_{k}"]
                max_k = content_range_body[f"max_{k}"]
                if f"min_{k}" in touch_body:
                    min_k = min(min_k, touch_body[f"min_{k}"])
                    max_k = max(max_k, touch_body[f"max_{k}"])

                window_normalizer[bucket][f"min_{k}"] = min_k
                window_normalizer[bucket][f"max_{k}"] = max_k

        with tqdm(total=frame.bucket.max(), leave=False) as pbar:
            cur_bucket = 0
            # Tracks the tampered keys in this particular window.
            touch_body = {}
            for t in tqdm(frame.itertuples(), leave=False):
                # If we've encountered a new bucket.
                if t.bucket != cur_bucket:
                    add_normalizer(cur_bucket, content_body, touch_body)
                    cur_bucket = t.bucket
                    # Reset the touch tracker.
                    touch_body = {}
                    pbar.update(1)

                for key in att_keys:
                    value = getattr(t, key)
                    if not np.isnan(value):
                        if t.optype == "data":
                            # If this is a data tuple, update the actual data tracker.
                            if value not in content_body[key]:
                                content_body[key][value] = t.freq_count
                            else:
                                content_body[key][value] += t.freq_count

                            # Remove the key from content_body if it's gone to 0.
                            if content_body[key][value] == 0:
                                del content_body[key][value]
                                content_range_body[f"min_{key}"] = min(content_body[key])
                                content_range_body[f"max_{key}"] = max(content_body[key])
                            else:
                                # Otherwise update the mutate tracker.
                                if f"min_{key}" not in content_range_body:
                                    content_range_body[f"min_{key}"] = value
                                    content_range_body[f"max_{key}"] = value
                                else:
                                    if content_range_body[f"min_{key}"] > value:
                                        content_range_body[f"min_{key}"] = value
                                    if content_range_body[f"max_{key}"] < value:
                                        content_range_body[f"max_{key}"] = value
                        else:
                            # Otherwise update the mutate tracker.
                            if f"min_{key}" not in touch_body:
                                touch_body[f"min_{key}"] = value
                                touch_body[f"max_{key}"] = value
                            else:
                                if touch_body[f"min_{key}"] > value:
                                    touch_body[f"min_{key}"] = value
                                if touch_body[f"max_{key}"] < value:
                                    touch_body[f"max_{key}"] = value

            add_normalizer(cur_bucket, content_body, touch_body)
            pbar.update(1)

        # Pad out the window_normalizer in a "state-preserving manner".
        for i in range(len(query_orders) + 1):
            if i not in window_normalizer:
                # Take the closest stats from the window before.
                j = i - 1
                while j not in window_normalizer and j > 0:
                    j -= 1
                assert j in window_normalizer
                window_normalizer[i] = window_normalizer[j]

        return data_frame, window_normalizer

    input_frame, window_normalizer = acquire_base_data()
    for optype, frame in tqdm(input_frame.groupby(by=["optype"]), leave=False):
        def generate_histogram(optype, index_clause, frame):
            maintenance_body = {k: {} for k in att_keys}
            frame = frame.sort_values(by=["bucket"], ignore_index=True)
            with tqdm(total=frame.bucket.max(), leave=False) as pbar:
                cur_bucket = 0
                for t in tqdm(frame.itertuples(), leave=False):
                    if t.bucket != cur_bucket:
                        if optype == "data":
                            # Forward the "data" state through the windows.
                            while cur_bucket < t.bucket:
                                create_histograms(cur_bucket, optype, index_clause, maintenance_body, normalizer=window_normalizer[cur_bucket])
                                cur_bucket += 1
                        elif optype != "data":
                            create_histograms(cur_bucket, optype, index_clause, maintenance_body, normalizer=window_normalizer[cur_bucket])
                            maintenance_body = {k: {} for k in att_keys}

                        cur_bucket = t.bucket
                        pbar.update(1)

                    for key in att_keys:
                        value = getattr(t, key)
                        if not np.isnan(value):
                            if value not in maintenance_body[key]:
                                maintenance_body[key][value] = t.freq_count
                            else:
                                maintenance_body[key][value] += t.freq_count

                # Create the histogram for the data.
                if optype == "data":
                    normalizer_hist_cache = []

                    # Forward the "data" state through the windows.
                    create_histograms(cur_bucket, optype, index_clause, maintenance_body, normalizer=window_normalizer[cur_bucket])
                    normalizer_hist_cache.append((window_normalizer[cur_bucket], output_tuples[-1]))
                    cur_bucket += 1

                    # Apply a savings optimization since this can potentially be somewhat expensive.
                    while cur_bucket < len(query_orders):
                        inserted = False
                        for (normalizer, hist) in normalizer_hist_cache:
                            if normalizer == window_normalizer[cur_bucket]:
                                # The normalizer is the same, our inputs are the same, so just copy.
                                hist = copy.deepcopy(hist)
                                hist[0] = cur_bucket
                                output_tuples.append(hist)
                                inserted = True
                                break

                        if not inserted:
                            # The normalizer is different, so re-compute with the new normalizer.
                            create_histograms(cur_bucket, optype, index_clause, maintenance_body, normalizer=window_normalizer[cur_bucket])
                            normalizer_hist_cache.append((window_normalizer[cur_bucket], output_tuples[-1]))

                        cur_bucket += 1
                elif optype != "data":
                    create_histograms(cur_bucket, optype, index_clause, maintenance_body, normalizer=window_normalizer[cur_bucket])
                    maintenance_body = {k: {} for k in att_keys}
                pbar.update(1)

        if optype == "data" or int(optype) != OpType.SELECT.value:
            generate_histogram(optype, None, frame)
        else:
            generate_histogram(optype, None, frame)
            for uval in frame.index_used.unique():
                # Adjust the SELECT based on the index used.
                generate_histogram(optype, uval, frame[frame.index_used == uval])

    columns = ["window_index", "optype", "index_clause", "att_name", "key_dist"]
    df = pd.DataFrame(output_tuples, columns=columns)
    # Unify the optype column type as a string.
    df["optype"] = df.optype.astype(str)
    return df


def construct_keyspaces(logger, connection, work_prefix, tbls, table_attr_map, window_index_map, buckets, callback_fn=None):
    datatypes = {}
    with connection.transaction():
        result = connection.execute("SELECT table_name, column_name, data_type FROM information_schema.columns")
        for record in result:
            tbl, att, dt = record[0], record[1], record[2]
            if not tbl in datatypes:
                datatypes[tbl] = {}
            datatypes[tbl][att] = dt

    bucket_ks = {}
    with connection.cursor() as cursor:
        for tbl in tbls:
            if isinstance(window_index_map[tbl], range) or isinstance(window_index_map[tbl], list):
                query_orders = window_index_map[tbl]
            else:
                # Assume it is a data frame then.
                query_orders = window_index_map[tbl].query_order.values

            # This logic will yoink histograms for all the attributes of the table.
            # And not exactly whether it is in the table keyspace or not.
            attrs = []
            for a in table_attr_map[tbl]:
                mod_tbl = f"{work_prefix}_{tbl}"
                if mod_tbl in datatypes and a in datatypes[mod_tbl]:
                    t = datatypes[mod_tbl][a]
                    # FIXME: We don't currently construct good histograms for textual data.
                    # However, you could probably linearize the textual data into an access distribution.
                    if not (t == "text" or "character" in t):
                        attrs.append(a)

            if len(attrs) == 0:
                # There aren't any attributes of interest.
                logger.info("Skipping querying keyspace distribution for %s", tbl)
                continue

            logger.info("Querying keyspace distribution from access and raw data (%s): %s (%s)", attrs, tbl, datetime.now())
            df = __execute_dist_query(logger, cursor, work_prefix, tbl, attrs, query_orders, buckets)

            if callback_fn is not None:
                df = callback_fn(tbl, df)
                if df is not None:
                    bucket_ks[tbl] = df
            else:
                bucket_ks[tbl] = df
    return bucket_ks
