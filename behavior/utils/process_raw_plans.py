import pglast
import pandas as pd
import numpy as np
import pandas as pd
import glob
import json

from behavior.model_workload.utils import OpType


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


def read_all_plans(input_dir):
    plans = glob.glob(f"{input_dir}/stats.*/pg_qss_plans_*.csv")
    def read_csv(file):
        data = pd.read_csv(file)
        data["query_id"] = data.query_id.astype(np.int64)
        return data
    return pd.concat(map(read_csv, plans), ignore_index=True)


def _process_qss_plans(logger, pg_qss_plans, block, extract_ous, table_attr_map=None, indexoid_table_map=None, indexoid_name_map=None):
    # Remove all invalid queries.
    pg_qss_plans = pg_qss_plans[pg_qss_plans.query_id != 0].copy()

    if not extract_ous:
        pg_qss_plans["query_text"] = ""
        pg_qss_plans["target"] = ""
        pg_qss_plans["num_rel_refs"] = 0
        pg_qss_plans["optype"] = 0
        pg_qss_plans["limit"] = 0
        pg_qss_plans["orderby"] = ""

    # Take a first pass through and populate any additional features.
    # Check also whether the query_text is valid or not.
    for plan in pg_qss_plans.itertuples():
        feature = json.loads(plan.features)
        if len(feature) > 1:
            if logger is not None:
                logger.warn("Skipping decoding of plan: %s", feature)
            continue

        # Get the query text and normalize it.
        query_text = feature[0]["query_text"].lower().rstrip()
        if not extract_ous:
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

            pg_qss_plans.at[plan.Index, "num_rel_refs"] = num_rel_refs
            pg_qss_plans.at[plan.Index, "target"] = target
            pg_qss_plans.at[plan.Index, "optype"] = optype
            pg_qss_plans.at[plan.Index, "limit"] = limit
            pg_qss_plans.at[plan.Index, "orderby"] = ",".join(orderbys)

        for query in block:
            if query_text is not None and query in query_text:
                query_text = None
        pg_qss_plans.at[plan.Index, "query_text"] = query_text

    new_df_tuples = []
    for row in pg_qss_plans.itertuples():
        if row.query_text is not None:
            def construct_plan_tuple(row, plan, target_plan_features, target_idx_scan_table, target_idx_scan):
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
                }

                if extract_ous:
                    new_tuple["features"] = target_plan_features
                else:
                    new_tuple.update({
                    'query_text': row.query_text,
                    'num_rel_refs': row.num_rel_refs,
                    'target': row.target,
                    'target_idx_scan_table': target_idx_scan_table,
                    'target_idx_scan': target_idx_scan,
                    'optype': row.optype,
                    'limit': row.limit,
                    'orderby': row.orderby,
                })

                new_df_tuples.append(new_tuple)

            def process_plan(plan):
                target_idx_scan_table = None
                target_idx_scan = None
                target_plan_features = None
                plan_features = {}
                for key, value in plan.items():
                    if key == "Plans":
                        # For the special key, we recurse into the child.
                        for p in value:
                            process_plan(p)
                        continue

                    if extract_ous:
                        if isinstance(value, list):
                            # FIXME(LIST): For now, we simply featurize a list[] with a numeric length.
                            # This is likely insufficient if the content of the list matters significantly.
                            key = key + "_len"
                            value = len(value)
                        plan_features[key] = value

                if extract_ous:
                    # Dump the plan features.
                    target_plan_features = json.dumps(plan_features)
                else:
                    # Directly try and yoink the relevant feature.
                    col = "IndexScan_indexid" if "IndexScan_indexid" in plan else "IndexOnlyScan_indexid"
                    if col in plan and plan[col] in indexoid_table_map:
                        target_idx_scan_table = indexoid_table_map[plan[col]]
                        target_idx_scan = indexoid_name_map[plan[col]]

                # Construct a plan row for the operator.
                construct_plan_tuple(row, plan, target_plan_features, target_idx_scan_table, target_idx_scan)

            if not extract_ous:
                # Construct one for the root -1.
                construct_plan_tuple(row, None, None, None, None)
            process_plan(json.loads(row.features)[0])

    return pd.DataFrame(new_df_tuples)


def process_raw_plans(logger, input_dir, block, extract_ous, table_attr_map=None, indexoid_table_map=None, indexoid_name_map=None):
    # Check the input arguments.
    if extract_ous:
        assert table_attr_map is None and indexoid_table_map is None and indexoid_name_map is None
    else:
        assert table_attr_map is not None and indexoid_table_map is not None and indexoid_name_map is not None

    plans = read_all_plans(input_dir)
    return _process_qss_plans(logger, plans, block, extract_ous, table_attr_map, indexoid_table_map, indexoid_name_map)
