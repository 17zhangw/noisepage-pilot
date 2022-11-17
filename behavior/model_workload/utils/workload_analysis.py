import glob
import json
import pglast
import pandas as pd
import numpy as np
from psycopg.rows import dict_row
from behavior.model_workload.utils import OpType
from behavior.utils import read_all_plans

class WorkloadAnalysis:
    # table_attr_map defines all attributes that might exist per table.
    table_attr_map = None
    # Here we construct all INDEX columns that are possibly of value.
    table_index_map = None
    # Relation OID -> Relation name.
    reloid_table_map = None
    # Defines the keyspace for a relation. This is taken as the INDEX space OR the PK space.
    table_keyspace_map = None
    # Index OID -> Table name.
    indexoid_table_map = None
    # Index OID -> Index name.
    indexoid_name_map = None
    # Query Text -> { (tbl, col) -> (tbl, col/arg) }
    query_template_map = None
    # (query text, qid, is_insert/update/delete) -> (target table)
    query_table_map = None
    # qid -> (limit, order by)
    query_sorts_map = None


# Queries to block.
QUERY_CONTENT_BLOCK = [
    "pg_",
    "version()",
    "current_schema()",
    "pgstattuple_approx",
]


def process_schemas(connection, input_dir, workload_only, tables):
    # table_attr_map defines all attributes that might exist per table.
    table_attr_map = {t: [] for t in tables}
    reloid_table_map = {}

    # Here we construct all INDEX columns that are possibly of value.
    # This is done by extracting all INDEXed key columns and then augmenting on IDX-JOINs.
    table_index_map = {t: set() for t in tables}
    # Defines the keyspace for a relation. This is taken as the INDEX space OR the PK space.
    # All tuple inserts/deletes are done relative to the PK space. INDEX space is used
    # for looking at data distribution w.r.t to index clustering.
    table_keyspace_map = {t: {} for t in tables}

    # Process pg_index and pg_class from files to build the indexoid_table_map.
    indexoid_table_map = {}
    indexoid_name_map = {}

    # FIXME(NON_SCHEMA): We assume that there aren't useful schema changes to be identified
    # by assuming that sample at t=0 for the schema is the same at any point in the data file.
    # Otherwise, we could arbitrarily split the analysis off at that point in time.
    with connection.cursor(row_factory=dict_row) as cursor:
        # Read in the files.
        pg_index = pd.read_csv(f"{input_dir}/pg_index.csv" if not workload_only else f"{input_dir}/pg_index.csv.0")
        pg_class = pd.read_csv(f"{input_dir}/pg_class.csv" if not workload_only else f"{input_dir}/pg_class.csv.0")
        pg_attribute = pd.read_csv(f"{input_dir}/pg_attribute.csv" if not workload_only else f"{input_dir}/pg_attribute.csv.0", low_memory=False)

        # Make the t=0 is same as t=end assumption.
        pg_index = pg_index[pg_index.time == pg_index.iloc[-1].time]
        pg_class = pg_class[pg_class.time == pg_class.iloc[-1].time]
        pg_attribute = pg_attribute[pg_attribute.time == pg_attribute.iloc[-1].time]
        for tup in pg_class.itertuples():
            reloid_table_map[f"{tup.oid}"] = tup.relname

        pg_index.set_index(keys=["indrelid"], inplace=True)
        pg_class.set_index(keys=["oid"], inplace=True)

        # First join the pg_index -> to pg_class to yoink the table.
        pg_index = pg_index.join(pg_class, how="inner", rsuffix="_class")
        pg_index.reset_index(drop=False, inplace=True)
        pg_index["indrelid"] = pg_index["index"]
        pg_index.drop(columns=["index"], inplace=True)

        # Then join the pg_index -> to pg_class to yoink the index.
        pg_index.set_index(keys=["indexrelid"], inplace=True)
        pg_index = pg_index.join(pg_class, how="inner", rsuffix="_index")
        pg_index.reset_index(drop=False, inplace=True)
        pg_index["indexrelid"] = pg_index["index"]
        pg_index.drop(columns=["index"], inplace=True)

        # Join the pg_attribute -> pg_class to form table/attrs.
        pg_attribute.set_index(keys=["attrelid"], inplace=True)
        pg_attribute = pg_attribute.join(pg_class, how="inner", rsuffix="_class")
        pg_attribute.reset_index(drop=False, inplace=True)
        for pgatt in pg_attribute.groupby(by=["relname"]):
            tbl, atts = pgatt[0], pgatt[1]
            atts.sort_values(by=["attnum"], inplace=True)
            if tbl in table_attr_map:
                table_attr_map[tbl] = atts[atts.attnum >= 1].attname.values

        for tup in pg_index.itertuples():
            # Construct the indexoid_table_map.
            indexoid_table_map[tup.indexrelid] = tup.relname
            indexoid_name_map[tup.indexrelid] = tup.relname_index

            attnums = tup.indkey.split(" ")
            table_keyspace_map[tup.relname][tup.relname_index] = []
            for attnum in attnums:
                df = (pg_attribute[(pg_attribute.relname == tup.relname) & (pg_attribute.attnum == int(attnum))])
                assert len(df) == 1

                att = df.iloc[0]
                if tup.relname in table_index_map:
                    table_index_map[tup.relname].add(att.attname)
                    table_keyspace_map[tup.relname][tup.relname_index].append(att.attname)

            if tup.indisprimary:
                # Handle the primary key for the table keyspace.
                table_keyspace_map[tup.relname][tup.relname] = table_keyspace_map[tup.relname][tup.relname_index]
    return table_attr_map, reloid_table_map, table_index_map, table_keyspace_map, indexoid_table_map, indexoid_name_map


def process_plans(input_dir, table_attr_map, indexoid_table_map, indexoid_name_map):
    # Process all the plans to get the query text.
    pg_qss_plans = read_all_plans(input_dir)
    pg_qss_plans["query_text"] = ""
    pg_qss_plans["target"] = ""
    pg_qss_plans["num_rel_refs"] = 0
    pg_qss_plans["optype"] = 0
    pg_qss_plans["limit"] = 0
    pg_qss_plans["orderby"] = ""
    for plan in pg_qss_plans.itertuples():
        feature = json.loads(plan.features)
        query_text = feature[0]["query_text"].lower().rstrip()

        target = ""
        num_rel_refs = 0
        optype = 0
        limit = 0
        orderbys = []
        if query_text is not None:
            root = pglast.Node(pglast.parse_sql(query_text))

            # First construct all the table aliases.
            alias_tbl_map = {}
            for node in root.traverse():
                if isinstance(node, pglast.node.Node):
                    if isinstance(node.ast_node, pglast.ast.RangeVar):
                        # Parse the alias map.
                        if node.ast_node.alias is not None:
                            alias_tbl_map[node.ast_node.alias.aliasname] = node.ast_node.relname
                        alias_tbl_map[node.ast_node.relname] = node.ast_node.relname

            for node in root.traverse():
                if isinstance(node, pglast.node.Node):
                    if isinstance(node.ast_node, pglast.ast.InsertStmt):
                        assert num_rel_refs == 0
                        target = node.ast_node.relation.relname
                        num_rel_refs = 1
                        optype = OpType.INSERT.value
                    elif isinstance(node.ast_node, pglast.ast.UpdateStmt):
                        assert num_rel_refs == 0
                        target = node.ast_node.relation.relname
                        num_rel_refs = 1
                        optype = OpType.UPDATE.value
                    elif isinstance(node.ast_node, pglast.ast.DeleteStmt):
                        assert num_rel_refs == 0
                        target = node.ast_node.relation.relname
                        num_rel_refs = 1
                        optype = OpType.DELETE.value
                    elif isinstance(node.ast_node, pglast.ast.SelectStmt) and node.ast_node.fromClause is not None:
                        optype = OpType.SELECT.value
                        for n in node.ast_node.fromClause:
                            if isinstance(n, pglast.ast.RangeVar):
                                num_rel_refs = num_rel_refs + 1
                                if len(target) == 0:
                                    target = n.relname
                                else:
                                    target = target + "," + n.relname

                        if node.ast_node.limitCount:
                            limit = node.ast_node.limitCount.val.val

                        if node.ast_node.sortClause:
                            for n in node.ast_node.sortClause:
                                sort = "DESC" if n.sortby_dir == pglast.enums.parsenodes.SortByDir.SORTBY_DESC else "ASC"
                                if isinstance(n.node, pglast.ast.ColumnRef):
                                    tbl, col = extract_tbl_col_pair(alias_tbl_map, table_attr_map, n.node.fields)
                                    orderbys.append(f"{tbl} {col} {sort}")

        for query in QUERY_CONTENT_BLOCK:
            if query_text is not None and query in query_text:
                query_text = None
        pg_qss_plans.at[plan.Index, "query_text"] = query_text
        pg_qss_plans.at[plan.Index, "num_rel_refs"] = num_rel_refs
        pg_qss_plans.at[plan.Index, "target"] = target
        pg_qss_plans.at[plan.Index, "optype"] = optype
        pg_qss_plans.at[plan.Index, "limit"] = limit
        pg_qss_plans.at[plan.Index, "orderby"] = ",".join(orderbys)

    new_df_tuples = []
    for row in pg_qss_plans.itertuples():
        feature = json.loads(row.features)
        if row.query_text is not None:
            def construct_plan_tuple(row, plan, target_idx_scan_table, target_idx_scan):
                new_tuple = {
                    'query_id': row.query_id,
                    'generation': row.generation,
                    'db_id': row.db_id,
                    'pid': row.pid,
                    'statement_timestamp': row.statement_timestamp,
                    'plan_node_id': -1 if plan is None else plan["plan_node_id"],

                    'left_child_node_id': plan["left_child_node_id"] if plan is not None and "left_child_node_id" in plan else -1,
                    'right_child_node_id': plan["right_child_node_id"] if plan is not None and "right_child_node_id" in plan else -1,
                    'total_cost': 0 if plan is None else plan["total_cost"],
                    'startup_cost': 0 if plan is None else plan["startup_cost"],

                    'query_text': row.query_text,
                    'num_rel_refs': row.num_rel_refs,
                    'target': row.target,
                    'target_idx_scan_table': target_idx_scan_table,
                    'target_idx_scan': target_idx_scan,
                    'optype': row.optype,
                    'limit': row.limit,
                    'orderby': row.orderby,
                }
                new_df_tuples.append(new_tuple)

            # We have a valid plan.
            def process_plan(plan):
                for key, value in plan.items():
                    if key == "Plans":
                        # For the special key, we recurse into the child.
                        for p in value:
                            process_plan(p)
                        continue

                # Directly try and yoink the relevant feature.
                col = "IndexScan_indexid" if "IndexScan_indexid" in plan else "IndexOnlyScan_indexid"
                target_idx_scan_table = None
                target_idx_scan = None
                if col in plan and plan[col] in indexoid_table_map:
                    target_idx_scan_table = indexoid_table_map[plan[col]]
                    target_idx_scan = indexoid_name_map[plan[col]]
                construct_plan_tuple(row, plan, target_idx_scan_table, target_idx_scan)

            # Let's append one for the -1. So we actually have something to match on to populate useful data.
            construct_plan_tuple(row, None, None, None)

            # Generate the plan tuples.
            process_plan(json.loads(row.features)[0])

    new_df_tuples = pd.DataFrame(new_df_tuples)
    new_df_tuples.set_index(keys=["statement_timestamp"], drop=True, append=False, inplace=True)
    new_df_tuples.sort_index(axis=0, inplace=True)
    return new_df_tuples


def extract_tbl_col_pair(alias_tbl_map, table_attr_map, columnref):
    tbl = None
    val = None
    if len(columnref) == 2:
        alias = columnref[0].val
        if alias in alias_tbl_map:
            tbl = alias_tbl_map[alias]
        val = columnref[1].val
    else:
        for t in alias_tbl_map:
            if t in table_attr_map and columnref[0].val in table_attr_map[t]:
                tbl = t
                break
        val = columnref[0].val
    return tbl, val


def process_query_templates(pg_qss_plans, table_attr_map):
    # Process each query and extract useful parameters w.r.t to INDEX keyspaces.
    query_template_map = {}
    for plan_group in pg_qss_plans.groupby(by=["query_text"]):
        query_template_map[plan_group[0]] = {}
        root = pglast.Node(pglast.parse_sql(plan_group[0]))

        # First construct all the table aliases.
        alias_tbl_map = {}
        for node in root.traverse():
            if isinstance(node, pglast.node.Node):
                if isinstance(node.ast_node, pglast.ast.RangeVar):
                    # Parse the alias map.
                    if node.ast_node.alias is not None:
                        alias_tbl_map[node.ast_node.alias.aliasname] = node.ast_node.relname
                    alias_tbl_map[node.ast_node.relname] = node.ast_node.relname

        for node in root.traverse():
            if isinstance(node, pglast.node.Node):
                if isinstance(node.ast_node, pglast.ast.InsertStmt):
                    values = node.ast_node.selectStmt.valuesLists[0]
                    tbl = [k for k in alias_tbl_map.keys()][0]
                    if node.ast_node.cols is None:
                        # Specific case of INSERT INTO [tbl] VALUES ()
                        assert node.ast_node.relation.relname in table_attr_map
                        for i, target in enumerate(table_attr_map[node.ast_node.relation.relname]):
                            column = target
                            param = values[i].number
                            query_template_map[plan_group[0]][(tbl, column)] = (None, f"arg{param}")
                    else:
                        # Specific case of INSERT INTO [tbl] (cols) VALUES ()
                        for i, target in enumerate(node.ast_node.cols):
                            column = target.name
                            param = values[i].number
                            query_template_map[plan_group[0]][(tbl, column)] = (None, f"arg{param}")
                elif isinstance(node.ast_node, pglast.ast.A_Expr):
                    # It can be of the form [column] [OP] [param]
                    # Or it can be of the form [param] [OP] [column]
                    column = None
                    param = None
                    ast = node.ast_node
                    if (isinstance(ast.lexpr, pglast.ast.ColumnRef) and isinstance(ast.rexpr, pglast.ast.ParamRef)):
                        column = ast.lexpr.fields
                        param = ast.rexpr.number
                    elif (isinstance(ast.lexpr, pglast.ast.ParamRef) and isinstance(ast.rexpr, pglast.ast.ColumnRef)):
                        column = ast.rexpr.fields
                        param = ast.lexpr.number

                    if column is not None:
                        tbl, column = extract_tbl_col_pair(alias_tbl_map, table_attr_map, column)
                        operator = node.ast_node.name[0].val
                        # Range clause.
                        if operator == "<":
                            column = column + "_high"
                        elif operator == ">=":
                            column = column + "_leq"
                        elif operator == "<=":
                            column = column + "_heq"
                        elif operator == ">":
                            column = column + "_low"
                        # This is an useful index column.
                        assert param is not None
                        query_template_map[plan_group[0]][(tbl, column)] = (None, f"arg{param}")

                    if (isinstance(ast.lexpr, pglast.ast.ColumnRef) and isinstance(ast.rexpr, pglast.ast.ColumnRef)):
                        ltbl, lcol = extract_tbl_col_pair(alias_tbl_map, table_attr_map, ast.lexpr.fields)
                        rtbl, rcol = extract_tbl_col_pair(alias_tbl_map, table_attr_map, ast.rexpr.fields)
                        # We don't want to indiscriminately overwrite in the case of s.id = $1 and s.id = c.id.
                        # This is the case where there is [a] column = [b] column.
                        if (ltbl, lcol) not in query_template_map[plan_group[0]]:
                            query_template_map[plan_group[0]][(ltbl, lcol)] = (rtbl, rcol)
                        if (rtbl, rcol) not in query_template_map[plan_group[0]]:
                            query_template_map[plan_group[0]][(rtbl, rcol)] = (ltbl, lcol)
    return query_template_map


def workload_analysis(connection, input_dir, workload_only, tables):
    # Process the schemas.
    table_attr_map, reloid_table_map, table_index_map, table_keyspace_map, indexoid_table_map, indexoid_name_map = process_schemas(connection, input_dir, workload_only, tables)

    # Process the plans.
    pg_qss_plans = process_plans(input_dir, table_attr_map, indexoid_table_map, indexoid_name_map)

    # This processes a mapping from query template -> ($1 -> "...").
    # As such, we identify a map from argument key to argument value per query template.
    query_template_map = process_query_templates(pg_qss_plans, table_attr_map)

    # Adjust the query templates and the attribute list.
    query_templates = query_template_map.keys()
    for key in query_templates:
        query_keys = {}
        for (ltbl, lkey), (rtbl, rkey) in query_template_map[key].items():
            orig_k = lkey
            if lkey.endswith("_high"):
                lkey = lkey[:-5]
            elif lkey.endswith("_leq"):
                lkey = lkey[:-4]
            elif lkey.endswith("_heq"):
                lkey = lkey[:-4]
            elif lkey.endswith("_low"):
                lkey = lkey[:-4]

            # The column is used in an INDEX.
            # We might need this column to specify another query.
            # Or is used as a reference.
            index_use = lkey in table_index_map[ltbl]
            ref_arg = any([(ltbl, lkey) in query_template_map[q] and not query_template_map[q][(ltbl, lkey)][1].startswith("arg") for q in query_template_map])
            ref_key = any([(rtbl, rkey) in query_template_map[q] for q in query_template_map])
            if index_use or ref_arg or ref_key:
                query_keys[(ltbl, orig_k)] = (rtbl, rkey)
        query_template_map[key] = query_keys

    for tbl in tables:
        old_attr_list = table_attr_map[tbl]
        new_attr_list = [a for a in old_attr_list
                           if any([a in table_index_map[t] for t in tables]) # Is the key part of any index
                           or any([(tbl, a) in query_template_map[b] for b in query_template_map]) # Is attr in the query template map.
                           or any([(tbl, a + "_high") in query_template_map[b] for b in query_template_map]) # Is attr in the query template map.
                           or any([(tbl, a + "_leq") in query_template_map[b] for b in query_template_map]) # Is attr in the query template map.
                           or any([(tbl, a + "_heq") in query_template_map[b] for b in query_template_map]) # Is attr in the query template map.
                           or any([(tbl, a + "_low") in query_template_map[b] for b in query_template_map])] # Is attr in the query template map.
        table_attr_map[tbl] = new_attr_list

    query_table_map = {(p.query_text, p.query_id, p.optype != OpType.SELECT.value): p.target for p in pg_qss_plans.itertuples()}
    query_sorts_map = {p.query_id: (p.limit, p.orderby) for p in pg_qss_plans.itertuples()}

    wa = WorkloadAnalysis()
    wa.table_attr_map = table_attr_map
    wa.table_index_map = table_index_map
    wa.reloid_table_map = reloid_table_map
    wa.table_keyspace_map = table_keyspace_map
    wa.indexoid_table_map = indexoid_table_map
    wa.indexoid_name_map = indexoid_name_map
    wa.query_template_map = query_template_map
    wa.query_table_map = query_table_map
    wa.query_sorts_map = query_sorts_map
    return wa, pg_qss_plans
