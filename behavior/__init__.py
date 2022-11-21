from __future__ import annotations
from behavior.datagen.pg_collector_utils import KNOBS
from enum import Enum

# A mapping from supported benchmark to tables used in the benchmark. This mapping is primarily
# used by doit behavior_pg_analyze_benchmark.
BENCHDB_TO_TABLES = {
    "tpcc": [
        "warehouse",
        "district",
        "customer",
        "item",
        "stock",
        "oorder",
        "history",
        "order_line",
        "new_order",
    ],
    "tatp": [
        "subscriber",
        "special_facility",
        "access_info",
        "call_forwarding",
    ],
    "tpch": [
        "region",
        "nation",
        "customer",
        "supplier",
        "part",
        "orders",
        "partsupp",
        "lineitem",
    ],
    "wikipedia": [
        "useracct",
        "watchlist",
        "ipblocks",
        "logging",
        "user_groups",
        "recentchanges",
        "page",
        "revision",
        "page_restrictions",
        "text",
    ],
    "voter": [
        "contestants",
        "votes",
        "area_code_state",
    ],
    "twitter": ["user_profiles", "tweets", "follows", "added_tweets", "followers"],
    "smallbank": ["accounts", "checking", "savings"],
    "sibench": ["sitest"],
    "seats": [
        "country",
        "airline",
        "airport",
        "customer",
        "flight",
        "airport_distance",
        "frequent_flyer",
        "reservation",
        "config_profile",
        "config_histograms",
    ],
    "resourcestresser": ["iotable", "cputable", "iotablesmallrow", "locktable"],
    "noop": ["fake"],
    "epinions": ["item", "review", "useracct", "trust", "review_rating"],
    "auctionmark": [
        "region",
        "useracct",
        "category",
        "config_profile",
        "global_attribute_group",
        "item",
        "item_comment",
        "useracct_feedback",
        "useracct_attributes",
        "item_bid",
        "useracct_watch",
        "global_attribute_value",
        "item_attribute",
        "item_image",
        "item_max_bid",
        "item_purchase",
        "useracct_item",
    ],
    "ycsb": ["usertable"],
}

BENCHDB_TO_INDEX = {
    "tpcc": [
        "warehouse_pkey",
        "district_pkey",
        "item_pkey",
        "new_order_pkey",
        "oorder_pkey",
        "customer_pkey",
        "stock_pkey",
        "order_line_pkey",
        "idx_customer_name",
        "idx_order",
        "oorder_o_w_id_o_d_id_o_c_id_o_id_key",
    ],
}


# This list must be kept up to date with the OU definitions in cmu-db/postgres.
class OperatingUnit(Enum):
    Agg = 0
    Append = 1
    BitmapAnd = 2
    BitmapHeapScan = 3
    BitmapIndexScan = 4
    BitmapOr = 5
    CteScan = 6
    CustomScan = 7
    ForeignScan = 8
    FunctionScan = 9
    Gather = 10
    GatherMerge = 11
    Group = 12
    Hash = 13
    HashJoinImpl = 14
    IncrementalSort = 15
    IndexOnlyScan = 16
    IndexScan = 17
    Limit = 18
    LockRows = 19
    Material = 20
    Memoize = 21
    MergeAppend = 22
    MergeJoin = 23
    ModifyTableInsert = 24
    ModifyTableUpdate = 25
    ModifyTableDelete = 26
    ModifyTableIndexInsert = 27
    NamedTuplestoreScan = 28
    NestLoop = 29
    ProjectSet = 30
    RecursiveUnion = 31
    Result = 32
    SampleScan = 33
    SeqScan = 34
    SetOp = 35
    Sort = 36
    SubPlan = 37
    SubqueryScan = 38
    TableFuncScan = 39
    TidScan = 40
    TidRangeScan = 41
    Unique = 42
    ValuesScan = 43
    WindowAgg =  44
    WorkTableScan = 45
    DestReceiverRemote = 46

    TupleARInsertTriggers = 47
    TupleARUpdateTriggers = 48
    TupleARDeleteTriggers = 49
    InsertUpdateFKTriggerEnforce = 50
    AfterTriggerEndQuery = 51
    DestReceiverSPI = 52
    InitIndexScan = 53
    InitIndexOnlyScan = 54
    InitLockRows = 55
    ExecutorStart = 56



class Targets(Enum):
    ELAPSED_US = "elapsed_us"


TARGET_COLUMNS = [n.value for n in Targets]


"""
A dictionary of derived features. The name is the name of the derived feature that follows the format
[OU]_[feature]. OU is guaranteed to be an unique identifier. The value of the derived feature is the
column in pg_qss_stats that should be remapped.
"""
EXECUTION_FEATURES_MAP = {
    "DestReceiverRemote_num_output": "counter0",

    "IndexOnlyScan_num_iterator_used": "counter0",
    "IndexOnlyScan_num_heap_fetches": "counter1",
    "IndexOnlyScan_num_defrag": "counter3",

    "IndexScan_num_iterator_used": "counter0",
    "IndexScan_num_heap_fetches": "counter1",
    "IndexScan_num_outer_loops": "counter2",
    "IndexScan_num_defrag": "counter3",

    "SeqScan_num_iterator_used": "counter0",
    "SeqScan_num_defrag": "counter1",

    "ModifyTableInsert_num_br_ir_as_triggers_fired": "counter0",
    "ModifyTableInsert_num_spec_insert": "counter1",
    "ModifyTableInsert_num_tuple_toast": "counter2",
    "ModifyTableInsert_num_fsm_checks": "counter3",
    "ModifyTableInsert_num_extends": "counter4",

    "ModifyTableIndexInsert_indexid": "payload",
    "ModifyTableIndexInsert_num_fsm_checks": "counter0",
    "ModifyTableIndexInsert_num_extends": "counter1",
    "ModifyTableIndexInsert_num_splits": "counter2",
    "ModifyTableIndexInsert_num_finish_splits": "counter3",

    "ModifyTableUpdate_num_br_ir_as_triggers_epq_fired": "counter0",
    "ModifyTableUpdate_num_index_updates_fired": "counter1",
    "ModifyTableUpdate_num_tuple_toast": "counter2",
    "ModifyTableUpdate_num_fsm_checks": "counter3",
    "ModifyTableUpdate_num_extends": "counter4",
    "ModifyTableUpdate_num_key_changes": "counter5",
    "ModifyTableUpdate_num_acquire_tuplock": "counter6",
    "ModifyTableUpdate_num_lock_wait_members": "counter7",
    "ModifyTableUpdate_num_updates": "counter8",
    "ModifyTableUpdate_num_aborts": "counter9",

    "ModifyTableDelete_num_br_ir_as_triggers_fired": "counter0",
    "ModifyTableDelete_num_recheck_quals": "counter1",
    "ModifyTableDelete_num_tuple_returns": "counter2",
    "ModifyTableDelete_num_acquire_tuplock": "counter3",
    "ModifyTableDelete_num_lock_wait_members": "counter4",
    "ModifyTableDelete_num_deletes": "counter5",

    "LockRows_num_marks": "counter0",
    "LockRows_num_invoke_epq": "counter1",
    "LockRows_num_update_find_latest": "counter2",
    "LockRows_num_check_tuplock_state": "counter3",
    "LockRows_num_acquire_tuplock": "counter4",
    "LockRows_num_lock_wait_members": "counter5",
    "LockRows_num_heap_lock_updated_tuple_rec_tuples": "counter6",
    "LockRows_num_heap_lock_updated_tuple_rec_wait_members": "counter7",
    "LockRows_num_non_lock_wait_block_policy": "counter8",
    "LockRows_num_aborts": "counter9",

    "Agg_num_input_rows": "counter0",

    "NestLoop_num_outer_rows": "counter0",
    "NestLoop_num_inner_rows_cumulative": "counter1",

    "BitmapIndexScan_num_tids_found": "counter0",

    "BitmapHeapScan_num_blocks_fetch": "counter0",
    "BitmapHeapScan_num_empty_tuples": "counter1",
    "BitmapHeapScan_num_tuples_fetch": "counter2",
    "BitmapHeapScan_num_blocks_prefetch": "counter3",
    "BitmapHeapScan_num_defrag": "counter4",

    "TupleARInsertTriggers_num_triggers": "counter0",
    "TupleARUpdateTriggers_num_triggers": "counter0",
    "TupleARDeleteTriggers_num_triggers": "counter0",

    "InsertUpdateFKTriggerEnforce_early_bail": "counter0",
    "InsertUpdateFKTriggerEnforce_prepare_plan": "counter1",
    "InsertUpdateFKTriggerEnforce_opt_generic": "counter2",
    "InsertUpdateFKTriggerEnforce_opt_custom": "counter3",

    "AfterTriggerEndQuery_num_invoke": "counter0",

    "DestReceiverSPI_num_output": "counter0",

    "InitIndexScan_num_scan_keys": "counter0",
    "InitIndexScan_num_runtime_keys": "counter1",
    "InitIndexScan_num_orderby_keys": "counter2",
    "InitIndexOnlyScan_num_scan_keys": "counter0",
    "InitIndexOnlyScan_num_runtime_keys": "counter1",
    "InitIndexOnlyScan_num_orderby_keys": "counter2",
    "InitLockRows_num_marks": "counter0",
    "ExecutorStart_num_marks": "counter0",
    "ExecutorStart_num_subplans": "counter1",
}


"""
This describes the map of derived input features to the models.
It is assumed that the key maps directly to the value.
"""
DERIVED_FEATURES_MAP = {k: k for k in EXECUTION_FEATURES_MAP}
DERIVED_FEATURES_MAP.update({"total_cost": "total_cost", "startup_cost": "startup_cost"})
DERIVED_FEATURES_MAP.update({"IndexScan_scan_est_pages_needed": "IndexScan_scan_est_pages_needed"})
DERIVED_FEATURES_MAP.update({"IndexOnlyScan_scan_est_pages_needed": "IndexOnlyScan_scan_est_pages_needed"})
DERIVED_FEATURES_MAP.update({"SeqScan_scan_est_pages_needed": "SeqScan_scan_est_pages_needed"})

for ou in OperatingUnit:
    # Generate per-OU switches for the KNOBs and the block stats.
    feats = ["blk_hit", "blk_miss", "blk_dirty", "blk_write"] + list(KNOBS.keys())
    DERIVED_FEATURES_MAP.update({ou.name + "_" + feat: feat for feat in feats})

    feats = ["total_cost", "startup_cost"]
    DERIVED_FEATURES_MAP.update({feat + "_" + ou.name: feat for feat in feats})
