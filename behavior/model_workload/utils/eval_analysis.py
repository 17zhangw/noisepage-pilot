
def load_eval_windows(logger, connection, work_prefix, max_arg, base_tbls, query_tables):
    with connection.transaction():
        base_tbls = set(base_tbls)
        hits_table_create = """
            CREATE UNLOGGED TABLE {prefix}_mw_eval_analysis (
                query_order BIGINT,
                query_id BIGINT,
                unix_timestamp BIGINT,
                optype INT,
                query_text TEXT,
                target TEXT,
                elapsed_us FLOAT8,
                {args},
                {tbl_hits})
            WITH (autovacuum_enabled = OFF)
        """.format(prefix=work_prefix,
                   args=",\n".join([f"arg{i+1} TEXT" for i in range(max_arg)]),
                   tbl_hits=",\n".join([f"{t}_hits INTEGER" for t in base_tbls]))
        connection.execute(hits_table_create)

        for tbl in set(query_tables):
            joins = []
            tbls = [t for t in tbl.split(",") if t in base_tbls]
            for t in tbls:
                joins.append("""
                    LEFT JOIN LATERAL (SELECT COUNT(1) as hits FROM {prefix}_{t}_hits {t} WHERE {t}.query_order = a.query_order) {t} ON true
                """.format(prefix=work_prefix, t=t))

            query = """
                INSERT INTO {prefix}_mw_eval_analysis (query_order, query_id, unix_timestamp, optype, query_text, target, elapsed_us, {args} {tbl_hits})
                SELECT a.query_order, a.query_id, a.unix_timestamp, a.optype, a.query_text, a.target, a.elapsed_us, {sel_args} {sel_tbl_hits}
                FROM {prefix}_mw_queries_args a
                {joins}
                WHERE a.target = '{tbl}'
            """.format(prefix=work_prefix,
                       args=",".join([f"arg{i+1}" for i in range(max_arg)]),
                       tbl_hits=("," + ",".join([f"{t}_hits" for t in tbls])) if len(tbls) > 0 else "",
                       sel_args=",".join([f"a.arg{i+1}" for i in range(max_arg)]),
                       sel_tbl_hits=("," + ",".join([f"{t}.hits" for t in tbls])) if len(tbls) > 0 else "",
                       joins="\n".join(joins),
                       tbl=tbl)
            logger.info("Executing load query for %s", tbl)
            logger.info("%s", query)
            connection.execute(query)

        logger.info("Creating index on the eval analysis table.")
        connection.execute(f"CREATE INDEX {work_prefix}_mw_eval_analysis_idx_qo ON {work_prefix}_mw_eval_analysis (query_order)")
        logger.info("Clustering the eval analysis table.")
        connection.execute(f"CLUSTER {work_prefix}_mw_eval_analysis USING {work_prefix}_mw_eval_analysis_idx_qo")
    connection.execute(f"VACUUM ANALYZE {work_prefix}_mw_eval_analysis")
