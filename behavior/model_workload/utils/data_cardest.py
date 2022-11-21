# Responsible for loading the initial shapshot data and computing deltas.
import os
import json
import pandas as pd
from pandas.api import types as pd_types

from behavior import OperatingUnit
from behavior.model_workload.utils import OpType


def get_type(datatypes, prefix, tbl, att):
    tbl = f"{prefix}_{tbl}"
    assert tbl in datatypes and att in datatypes[tbl]
    ty = datatypes[tbl][att]
    if "int" in ty:
        return "int"
    elif ty == "float" or "double" in ty:
        return "float8"
    else:
        assert ty == "text" or "character" in ty
        return "text"


def get_datatypes(connection):
    # This is not elegant and somewhat brittle.
    datatypes = {}
    with connection.transaction():
        result = connection.execute("SELECT table_name, column_name, data_type FROM information_schema.columns")
        for record in result:
            tbl, att, dt = record[0], record[1], record[2]
            if not tbl in datatypes:
                datatypes[tbl] = {}
            datatypes[tbl][att] = dt

    return datatypes


def load_initial_data(logger, connection, workload_only, work_prefix, input_dir, table_attr_map, table_keyspace_map):
    # Load the data.
    pg_class = pd.read_csv(f"{input_dir}/pg_class.csv" if not workload_only else f"{input_dir}/pg_class.csv.0")
    pg_attribute = pd.read_csv(f"{input_dir}/pg_attribute.csv" if not workload_only else f"{input_dir}/pg_attribute.csv.0")

    with connection.transaction():
        for tbl in table_attr_map.keys():
            if len(table_attr_map[tbl]) == 0:
                continue

            # We require that a PK exists. If a PK doesn't exist, then there is nothing worth looking at.
            # We use the PK for determining how data within a table evolves over time.
            if tbl not in table_keyspace_map[tbl]:
                continue

            logger.info("Loading data %s (%s)", tbl, table_attr_map[tbl])
            assert (input_dir / f"{tbl}_snapshot.csv").exists()
            input_frame = pd.read_csv(f"{input_dir}/{tbl}_snapshot.csv", usecols=table_attr_map[tbl], dtype=object)
            input_frame["insert_version"] = 0
            input_frame["delete_version"] = pd.NA
            input_frame["delete_version"] = input_frame.delete_version.astype("Int32")
            input_frame.to_csv(f"/tmp/{work_prefix}_{tbl}.csv", index=False)

            cls = pg_class[pg_class.relname == tbl].iloc[0]

            # This isn't efficient by any means...
            sql = f"CREATE UNLOGGED TABLE {work_prefix}_{tbl} ("
            fields = []
            for col in input_frame.columns:
                # Use the pg_attribute component to determine if it is a string or not.
                att = pg_attribute[(pg_attribute.attrelid == cls.oid) & (pg_attribute.attname == col)]
                if att.shape[0] > 0 and att.iloc[0].attlen == -1:
                    # This is a varying text field.
                    fields.append((col, 'text'))
                    continue

                try:
                    # Unfortunately we have to try and convert since I don't want to parse the pg_attribute column.
                    # And try to match a pgttypid.
                    input_frame[col] = input_frame[col].astype(int)
                    fields.append((col, 'integer'))
                    continue
                except:
                    pass

                try:
                    input_frame[col] = input_frame[col].astype(float)
                    fields.append((col, 'float8'))
                    continue
                except:
                    pass

                # This is a varying text field.
                fields.append((col, 'text'))
            sql += ",".join([f"{t[0]} {t[1]}" for t in fields])
            sql += ") WITH (autovacuum_enabled = OFF)"
            logger.info("Executing SQL: %s", sql)
            connection.execute(sql)

            sql = f"COPY {work_prefix}_{tbl} FROM '/tmp/{work_prefix}_{tbl}.csv' WITH (FORMAT csv, HEADER true, NULL '')"
            logger.info("Executing SQL: %s", sql)
            connection.execute(sql)
            os.remove(f"/tmp/{work_prefix}_{tbl}.csv")

        # Now build the index that we will use later.
        # Build for all the keyspaces.
        for tbl in table_keyspace_map:
            if len(tbl) == 0 or tbl not in table_keyspace_map[tbl]:
                continue

            cnt = 0
            installed_ks = set()
            for _, ks in table_keyspace_map[tbl].items():
                if tuple(ks) in installed_ks:
                    continue

                sql = f"CREATE INDEX {work_prefix}_{tbl}_{cnt} ON {work_prefix}_{tbl} ("
                sql += ",".join(ks) + ", insert_version)"
                logger.info("Executing SQL: %s", sql)
                connection.execute(sql)
                installed_ks.add(tuple(ks))
                cnt += 1

            if tbl in table_attr_map:
                for k in table_attr_map[tbl]:
                    # Build it on the single tuple too if needed.
                    if tuple([k]) not in installed_ks:
                        sql = f"CREATE INDEX {work_prefix}_{tbl}_{cnt} ON {work_prefix}_{tbl} ({k}, insert_version)"
                        logger.info("Executing SQL: %s", sql)
                        connection.execute(sql)
                        installed_ks.add(tuple(k))
                        cnt += 1

            connection.execute(f"CREATE INDEX {work_prefix}_{tbl}_iv ON {work_prefix}_{tbl} (insert_version)")

    for tbl in table_keyspace_map:
        if len(tbl) == 0 or tbl not in table_keyspace_map[tbl]:
            continue

        # Vacuum and analyze all the tables.
        connection.execute(f"VACUUM ANALYZE {work_prefix}_{tbl}")
    connection.execute(f"VACUUM ANALYZE {work_prefix}_mw_queries_args")
    connection.execute(f"VACUUM ANALYZE {work_prefix}_mw_queries")


def compute_data_change_frames(logger, connection, work_prefix, wa):
    table_attr_map = wa.table_attr_map
    table_keyspace_map = wa.table_keyspace_map
    query_template_map = wa.query_template_map
    query_table_map = wa.query_table_map
    datatypes = get_datatypes(connection)

    for (query, qid, ismod), tbl in query_table_map.items():
        if not ismod:
            continue

        if len(tbl.split(",")) != 1:
            # FIXME(INSUPDEL): We assume that INSERT/UPDATE/DELETE can only have 1 table reference. In practice,
            # that holds for postgres since there is only 1 target table.
            continue

        if tbl not in table_keyspace_map or tbl not in table_keyspace_map[tbl]:
            # FIXME(INSUPDEL): We make the required assumption that the table has a PK that we can exploit.
            # The PK is needed in order to help ensure that we have all the information needed to somewhat
            # guestimate the state of tuples.
            continue

        pk_keys = table_keyspace_map[tbl][tbl]
        assert len(pk_keys) > 0
        assert tbl in table_attr_map
        all_keys = table_attr_map[tbl]

        # FIXME(INSUPDEL): This assumes that query_order deterministically controls the visibility.
        # This is effectively implementing an explicit versioning control scheme where insert_version
        # and delete_version control when a slot is visible. In multi-concurrency setups, the query_order
        # is only an approximation of when a query might be visible.

        useful_args = [(k,v) for (_, k), (_, v) in query_template_map[query].items() if k in pk_keys or k in all_keys]
        if query.lower().startswith("insert"):
            query = "SELECT " + ",".join([f"{arg}::{get_type(datatypes, work_prefix, tbl, att)} as {att}" for att, arg in useful_args])
            query += f", query_order from {work_prefix}_mw_queries_args where optype = {OpType.INSERT.value} and target = '{tbl}' and query_id = {qid}"
            query = f"INSERT INTO {work_prefix}_{tbl} (" + ",".join([k for k,v in useful_args]) + ",insert_version) " + query
            logger.info("Executing SQL: %s", query)
            c = connection.execute(query)
            logger.info("Finished executing with affected rowcount: %s\n", c.rowcount)

        if query.lower().startswith("delete"):
            query = "SELECT " + ",".join([f"{arg}::{get_type(datatypes, work_prefix, tbl, att)} as {att}" for att, arg in useful_args])
            query += f", query_order from {work_prefix}_mw_queries_args where optype = {OpType.DELETE.value} and target = '{tbl}' and query_id = {qid}"
            query = f"UPDATE {work_prefix}_{tbl} as t SET delete_version = q.query_order FROM (" + query
            query += f") q WHERE t.insert_version < q.query_order and " + " and ".join([f"q.{k} = t.{k}" for (k,v) in useful_args])
            logger.info("Executing SQL: %s", query)
            c = connection.execute(query)
            logger.info("Finished executing with affected rowcount: %s\n", c.rowcount)
        
        # FIXME(UPDATE): Assume UPDATEs do not effect the primary keys of the tuples that they change.
        # We actually make a larger statement: UPDATEs don't change ANY keys whatsoever and so
        # don't require index inserts if a HOT is determined to be valid. Else we need both the
        # "old" keys as DELETE and the "new" keys as INSERT.


def probe_single(logger, connection, work_prefix, wa, datatypes, query, qid, query_tbl, plan_generation, pid, target, allowed_tbls):
    # These arguments are a little strange.
    #
    # query_tbl: references the original target that we need to use to filter mw_queries_args.
    # This is tied to what the query is doing and is not related to the "hits" being analyzed.
    #
    # target: describes the table's hits that we are analyzing. This is always a single table
    # reference even if the original query is a multi-way join.
    #
    # allowed_tbls: describes the set of tables on which we are allowed to consider predicates.
    # For fully-specified hits, this *must* be empty; for single-tables, this is [target].
    # for multi-way joins, this is the "information-channel" from the left.
    query_template_map = wa.query_template_map
    table_attr_map = wa.table_attr_map
    query_sorts_map = wa.query_sorts_map

    predicates = []
    use_tbls = set([target])
    rel_keys = table_attr_map[target]
    allowed_set_tbls = set(allowed_tbls)

    # Consider all the predicates that are potentially valid.
    for (ltbl, lkey), (rtbl, rkey) in query_template_map[query].items():
        norm_lkey = lkey
        if lkey.endswith("_high"):
            norm_lkey = lkey[:-5]
        elif lkey.endswith("_leq"):
            norm_lkey = lkey[:-4]
        elif lkey.endswith("_heq"):
            norm_lkey = lkey[:-4]
        elif lkey.endswith("_low"):
            norm_lkey = lkey[:-4]

        if ltbl not in table_attr_map and norm_lkey not in table_attr_map[ltbl]:
            # uh-oh
            continue

        if not ltbl in allowed_set_tbls:
            # We are not allowed to consider this argument.
            continue

        # FIXME(TABLE/INDEX): This is actually pretty awkward. There is a small disconnect between the filters
        # that are available and valid index predicates. In some cases, the "index" probe results in more
        # tuples because the index itself is more restrictive --.
        #
        # Those cases, you'll have two different numbers: (1) how many the index fetches and (2) how many
        # the scan outputs - which are the "hits" to the higher level.

        if not norm_lkey.startswith("arg") and not rkey.startswith("arg"):
            if rtbl in allowed_tbls:
                assert not rkey.endswith("_high") and not rkey.endswith("_loweq")
                predicates.append(f" {work_prefix}_{ltbl}.{norm_lkey} = {work_prefix}_{rtbl}.{rkey} ")
                use_tbls.add(rtbl)
            elif norm_lkey == lkey and rkey in table_attr_map[rtbl] and query_template_map[query][(rtbl, rkey)][1].startswith("arg"):
                # This is the case where we have ltbl.lkey = rtbl.rkey AND rtbl.rkey = $1.
                # FIXME(JOIN): This does seem like a hack because not all systems will do transitivity.
                other_arg = query_template_map[query][(rtbl, rkey)][1]
                ty = get_type(datatypes, work_prefix, ltbl, norm_lkey)
                predicates.append(f" {work_prefix}_{ltbl}.{norm_lkey} = a.{other_arg}::{ty} ")
                use_tbls.add(ltbl)
        elif lkey.endswith("_high") and rkey.startswith("arg"):
            assert rtbl is None
            ty = get_type(datatypes, work_prefix, ltbl, norm_lkey)
            predicates.append(f" {work_prefix}_{ltbl}.{norm_lkey} < a.{rkey}::{ty} ")
            use_tbls.add(ltbl)
        elif lkey.endswith("_leq") and rkey.startswith("arg"):
            assert rtbl is None
            ty = get_type(datatypes, work_prefix, ltbl, norm_lkey)
            predicates.append(f" {work_prefix}_{ltbl}.{norm_lkey} >= a.{rkey}::{ty} ")
            use_tbls.add(ltbl)
        elif lkey.endswith("_heq") and rkey.startswith("arg"):
            assert rtbl is None
            ty = get_type(datatypes, work_prefix, ltbl, norm_lkey)
            predicates.append(f" {work_prefix}_{ltbl}.{norm_lkey} <= a.{rkey}::{ty} ")
            use_tbls.add(ltbl)
        elif lkey.endswith("_low") and rkey.startswith("arg"):
            assert rtbl is None
            ty = get_type(datatypes, work_prefix, ltbl, norm_lkey)
            predicates.append(f" {work_prefix}_{ltbl}.{norm_lkey} > a.{rkey}::{ty} ")
            use_tbls.add(ltbl)
        elif rkey.startswith("arg"):
            assert rtbl is None
            ty = get_type(datatypes, work_prefix, ltbl, norm_lkey)
            predicates.append(f" {work_prefix}_{ltbl}.{norm_lkey} = a.{rkey}::{ty} ")
            use_tbls.add(ltbl)

    limit, orderby = query_sorts_map[qid]
    orderbys = orderby.split(",")
    sort_clauses = []
    if limit > 0:
        for clause in orderbys:
            s = clause.split(" ")
            tbl, col, order = s[0], s[1], s[2]
            if tbl in table_attr_map and col in table_attr_map[tbl] and tbl in use_tbls and tbl in allowed_set_tbls:
                # We are allowed to try and use the order by to enforce the query.
                # The interesting observation is that in the case we have exact args, allowed_tbls = [].
                # That means we don't care about the clause at all!
                sort_clauses.append(f"{work_prefix}_{tbl}.{col}")
    lateral = len(sort_clauses) > 0 and limit > 0

    # FIXME(BITMAP): Support resolving bitmap index scan use.
    idx_use_join = """
        LEFT JOIN LATERAL (SELECT query_order, target_idx_scan FROM {work_prefix}_mw_queries q WHERE
                                                a.query_id = q.query_id AND q.target = '{query_tbl}' AND
                                                a.query_order = q.query_order AND
                                                q.comment IN ('IndexScan', 'IndexOnlyScan') AND
                                                q.target_idx_scan_table = '{target}' AND
                                                q.plan_node_id >= 0
        ) q ON q.query_order = a.query_order
    """.format(work_prefix=work_prefix, query_tbl=query_tbl, target=target)

    q = f"INSERT INTO {work_prefix}_{target}_hits (query_order, statement_timestamp, unix_timestamp, optype, index_used, " + ",".join(rel_keys) + ") "
    # We will always extract these from queries_args table.
    q += f"SELECT a.query_order, a.statement_timestamp, a.unix_timestamp, a.optype, q.target_idx_scan"
    if len(allowed_tbls) == 0:
        assert len(sort_clauses) == 0
        assert plan_generation is None and pid is None
        # We extract the fields directly from mw_queries_args.
        args = []
        for k in rel_keys:
            rtbl, v = query_template_map[query][(query_tbl, k)]
            assert rtbl is None
            ty = get_type(datatypes, work_prefix, query_tbl, k)
            args.append(f"a.{v}::{ty}")
        q += "," + ",".join(args)
        q += f" FROM {work_prefix}_mw_queries_args a " + idx_use_join
        q += f" WHERE a.target = '{query_tbl}' and a.query_id = {qid}"
    else:
        # Attach the correct table prefix depending on if we need to lateral join or not.
        tbl_prefix = "b." if lateral else f"{work_prefix}_{target}."
        q += "," + ",".join([f"{tbl_prefix}{k}" for k in rel_keys])
        q += f" FROM {work_prefix}_mw_queries_args a "
        if lateral:
            # If we are performing a lateral, start the lateral subquery.
            q += "JOIN LATERAL (SELECT " + ",".join(rel_keys) + " FROM "
            # Insert all the relevant tables we are going to pull data from.
            q += ",".join([f"{work_prefix}_{t}" for t in use_tbls]) + " WHERE "

            # Attach all the predicates that we care about.
            q += " and ".join([f" {work_prefix}_{t}.insert_version <= a.query_order " for t in use_tbls])
            q += " and " + " and ".join([f" ({work_prefix}_{t}.delete_version is NULL or {work_prefix}_{t}.delete_version >= a.query_order) " for t in use_tbls])
            if len(predicates) > 0:
                q += " and " + " and ".join(predicates)

            # Attach the lateral sort.
            q += " ORDER BY " + ",".join(sort_clauses) + f" LIMIT {limit} "
            q += ") b ON true "
            q += idx_use_join + " WHERE "
        else:
            # Insert all the relevant tables we are going to pull data from.
            join_clause = """
                JOIN {work_prefix}_{t} ON {work_prefix}_{t}.insert_version <= a.query_order
                                      AND ({work_prefix}_{t}.delete_version IS NULL OR {work_prefix}_{t}.delete_version >= a.query_order)
                                      AND {aux_preds}
            """
            dual_preds = [pred for pred in predicates if pred.count(work_prefix) > 1]
            single_preds = [pred for pred in predicates if pred.count(work_prefix) == 1]
            table_preds = {t: [p for p in single_preds if t in p] for t in use_tbls}
            q += "\n".join([join_clause.format(work_prefix=work_prefix, t=t, aux_preds=" AND ".join(table_preds[t])) for t in use_tbls])
            q += idx_use_join + " WHERE "
            if len(dual_preds) > 0:
                q += " and ".join(dual_preds)
                q += " and "

        q += f" a.target = '{query_tbl}' and a.query_id = {qid} "
        if plan_generation is not None:
            q += f" and a.generation = {plan_generation}"

        if pid is not None:
            q += f" and a.pid = {pid}"

    logger.info(q)
    c = connection.execute(q, prepare=False)
    logger.info("Finished executing with: %s", c.rowcount)


def compute_underspecified(logger, connection, work_prefix, wa, plans):
    table_attr_map = wa.table_attr_map
    reloid_table_map = wa.reloid_table_map
    table_keyspace_map = wa.table_keyspace_map
    query_template_map = wa.query_template_map
    query_table_map = wa.query_table_map
    query_sorts_map = wa.query_sorts_map
    datatypes = get_datatypes(connection)

    hits_created = []
    with connection.transaction() as txn:
        for t in table_attr_map:
            tbl = f"{work_prefix}_{t}"
            if tbl in datatypes:
                # Create the _hits table.
                att_keys = [k for k in datatypes[tbl].keys() if k not in ["insert_version", "delete_version"]]
                atts = [
                    "query_order bigint",
                    "statement_timestamp bigint",
                    "unix_timestamp float8",
                    "optype int",
                    "index_used text",
                ] + [f"{k} {get_type(datatypes, work_prefix, t, k)}" for k in att_keys]
                tbl += "_hits"
                q = f"CREATE UNLOGGED TABLE {tbl} (" + ",".join(atts) + ") WITH (autovacuum_enabled = OFF)"
                connection.execute(q)
                hits_created.append(tbl)

        for (query, qid, ismod), query_tbl in query_table_map.items():
            tbls = [t for t in query_tbl.split(",") if t in table_keyspace_map]
            missing_args = set()
            all_keys = set()

            # This is to compute the keys that we are missing.
            # There is an interesting space complexity issue of whether we should precompute all of these or not.
            for interact_tbl in tbls:
                # Look through all keyspaces of the table.
                for _, ks in table_keyspace_map[interact_tbl].items():
                    # Look through constituent attributes in the keyspace.
                    for k in ks:
                        if (interact_tbl, k) not in query_template_map[query] or not query_template_map[query][(interact_tbl, k)][1].startswith("arg"):
                            # In this case, we aren't specifying [interact_tbl].[k] in the query or it's a dependent value.
                            missing_args.add((interact_tbl, k))
                all_keys.update([(interact_tbl, k) for _, ks in table_keyspace_map[interact_tbl].items() for k in ks])

            if len(all_keys) == 0:
                # This means that we don't actually have anything of interest to say.
                continue

            # FIXME(INPUT_KEY): We currently try to compute where in the keyspace we might end up touching. Perhaps the other
            # featurization that could work is we indicate that we have no idea what the space looks like! That way we can
            # communicate the underspecification of the query; but not sure if we should do that to begin with.

            if len(missing_args) == 0 and len(tbls) == 1:
                probe_single(logger, connection, work_prefix, wa, datatypes, query, qid, query_tbl, None, None, tbls[0], [])
            elif len(missing_args) > 0 and len(tbls) == 1:
                # We have missing arguments but we are only hitting a single table.
                probe_single(logger, connection, work_prefix, wa, datatypes, query, qid, query_tbl, None, None, tbls[0], [tbls[0]])
            elif len(missing_args) > 0:
                assert len(tbls) > 1

                # We look at the plan. And try to estimate based on what we know about the plan what estimates
                # we should actually look at. Observe that if there is a HASH, then we want to estimate
                # touched tuples independently; and if there is Nest, we want to estimate inner conditional
                # on the outer node.
                #
                # In theory we can batch this; effectively. you just need to make groups based on whether
                # the plan structure is the same or whether they have changed. And if they have changed,
                # welp you start a new group; since we have guarantees that generation is strictly
                # increasing over time.
                valid_plans = plans[plans.query_id == qid]
                for plan in valid_plans.itertuples():
                    features = json.loads(plan.features)
                    if features is None or len(features) == 0:
                        continue

                    def process_plan(current_node, carry):
                        nt = current_node["node_type"] if "node_type" in current_node else None
                        child = current_node["Plans"] if "Plans" in current_node else []
                        if nt == OperatingUnit.HashJoinImpl.name:
                            assert "HashJoinImpl not currently implemented"
                        elif nt == OperatingUnit.NestLoop.name:
                            assert len(child) == 2
                            # Assume that left is outer table so just process that as usual.
                            process_plan(child[0], carry)
                            # Push the carried information forwards!
                            process_plan(child[1], carry)
                        elif nt in [OperatingUnit.IndexScan.name, OperatingUnit.IndexOnlyScan.name, OperatingUnit.SeqScan.name]:
                            # FIXME(BITMAP): Support probing with the BitmapIndexScan.
                            key = {
                                OperatingUnit.IndexScan.name: "IndexScan_scan_scanrelid_oid",
                                OperatingUnit.IndexOnlyScan.name: "IndexOnlyScan_scan_scanrelid_oid",
                                OperatingUnit.SeqScan.name: "SeqScan_scanrelid_oid",
                            }[nt]

                            relname = reloid_table_map[f"{current_node[key]}"]
                            probe_single(logger, connection, work_prefix, wa, datatypes, query, qid, query_tbl, plan.generation, plan.pid, relname, [relname] + carry)
                            carry.append(relname)
                        else:
                            # Recurse into the children.
                            for c in child:
                                process_plan(c, carry)

                    process_plan(features[0], [])

    # Issue VACUUM.
    for tbl in hits_created:
        connection.execute(f"CREATE INDEX {tbl}_qo_idx ON {tbl} (query_order)")
        connection.execute(f"VACUUM ANALYZE {tbl}")
    connection.execute(f"VACUUM ANALYZE {work_prefix}_mw_queries_args")
    connection.execute(f"VACUUM ANALYZE {work_prefix}_mw_queries")
