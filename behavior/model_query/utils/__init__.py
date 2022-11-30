import copy
import pickle

class OUGenerationContext:
    # Window Index that the context belongs to.
    window_index = None
    # Lower bound of the query order in window.
    current_qo = None
    # Upper exclusive bound of the query order in window.
    upper_qo = None

    # List of relevant tables.
    tables = None
    # Mapping of table name -> table attributes.
    table_attr_map = None
    # Mapping of table name -> table feature state.
    table_feature_state = None
    # Mapping of pg_trigger_oid -> trigger information.
    trigger_info_map = None
    # Mapping of OID -> table name.
    oid_table_map = None
    # Mapping of index OID -> index feature state.
    index_feature_state = None
    # Mapping of indexoid -> table name.
    indexoid_table_map = None
    # Mapping of table -> index oids.
    table_indexoid_map = None
    # Relevant knobs (like shared buffers).
    shared_buffers = None

    # This is the state that should be fetched by each worker (such as models).
    ou_models = None
    table_feature_model = None
    table_state_model = None
    buffer_page_model = None
    buffer_access_model = None
    concurrency_model = None
    # Mapping of table to keyspace features.
    table_keyspace_features = None


    def save_state(self, window_index, current_qo, upper_qo):
        return copy.deepcopy({
            "window_index": window_index,
            "current_qo": current_qo,
            "upper_qo": upper_qo,

            "tables": self.tables,
            "table_attr_map": self.table_attr_map,
            "table_feature_state": self.table_feature_state,
            "trigger_info_map": self.trigger_info_map,
            "oid_table_map": self.oid_table_map,
            "index_feature_state": self.index_feature_state,
            "indexoid_table_map": self.indexoid_table_map,
            "table_indexoid_map": self.table_indexoid_map,
            "shared_buffers": self.shared_buffers,
        })


    def restore_state(self, state):
        self.window_index = state["window_index"]
        self.current_qo = state["current_qo"]
        self.upper_qo = state["upper_qo"]

        self.tables = state["tables"]
        self.table_attr_map = state["table_attr_map"]
        self.table_feature_state = state["table_feature_state"]
        self.trigger_info_map = state["trigger_info_map"]
        self.oid_table_map = state["oid_table_map"]
        self.index_feature_state = state["index_feature_state"]
        self.indexoid_table_map = state["indexoid_table_map"]
        self.table_indexoid_map = state["table_indexoid_map"]
        self.shared_buffers = state["shared_buffers"]

##################################################################################
# Load the Models
##################################################################################

def load_ou_models(path):
    model_dict = {}
    for model_path in path.rglob('*.pkl'):
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        model_dict[model.ou_name] = model
    return model_dict


def load_model(path, name):
    if path is None:
        return None

    import behavior.model_workload.models as model_workload_models
    model_cls = getattr(model_workload_models, name)
    return model_cls.load_model(path)
