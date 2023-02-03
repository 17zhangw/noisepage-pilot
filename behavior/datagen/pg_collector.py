import signal
from plumbum import cli
import pandas as pd
import binascii
import time
import csv
import argparse
import logging
import multiprocessing as mp
import re
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from distutils import util
from enum import Enum, auto, unique

from datetime import timedelta
from datetime import datetime

try:
    from bcc import BPF, USDT
except:
    pass

import psutil
import setproctitle
import psycopg
from psycopg.rows import dict_row
from behavior import BENCHDB_TO_TABLES, BENCHDB_TO_INDEX
from behavior.datagen.pg_collector_utils import SettingType, _time_unit_to_ms, _parse_field, KNOBS


logger = logging.getLogger("collector")
collector_pids = []


def lost_something(num_lost):
    # num_lost. pylint: disable=unused-argument
    pass


##################################################################3
# USER SPACE COLLECTOR
##################################################################3

# Name of output file/target --> (query, frequent)
PG_COLLECTOR_TARGETS = {
    "pg_stats": "SELECT EXTRACT(epoch from NOW())*1000000 as time, s.*, c.data_type FROM pg_stats s JOIN information_schema.columns c ON s.tablename=c.table_name AND s.attname=c.column_name WHERE schemaname = 'public';",
    "pg_class": "SELECT EXTRACT(epoch from NOW())*1000000 as time, * FROM pg_class t JOIN pg_namespace n ON n.oid = t.relnamespace WHERE n.nspname = 'public';",
    "pg_index": "SELECT EXTRACT(epoch from NOW())*1000000 as time, * FROM pg_index;",
    "pg_attribute": "SELECT EXTRACT(epoch from NOW())*1000000 as time, * FROM pg_attribute;",
    "pg_stat_user_tables": "SELECT EXTRACT(epoch from NOW())*1000000 as time, * FROM pg_stat_user_tables;",
    "pg_trigger": "SELECT EXTRACT(epoch from NOW())*1000000 as time, * FROM pg_trigger t, pg_constraint c WHERE t.tgconstraint = c.oid;",
}


def pg_collector(benchmark, output_rows, output_columns, slow_time, shutdown):
    def scrape_settings(connection, rows):
        result = []
        with connection.cursor(row_factory=dict_row) as cursor:
            tns = time.time_ns() / 1000
            cursor.execute("SHOW ALL;")
            for record in cursor:
                setting_name = record["name"]
                if setting_name in rows:
                    setting_type = rows[setting_name]
                    setting_str = record["setting"]
                    result.append((setting_name, _parse_field(setting_type, setting_str)))

        connection.commit()
        result.append(("time", tns))
        result.sort(key=lambda t: t[0])
        return result

    def scrape_table(connection, query):
        # Open a cursor to perform database operations
        tuples = []
        columns = []
        with connection.cursor() as cursor:
            # Query the database and obtain data as Python objects.
            cursor.execute(query, prepare=False)
            binary = []
            for i, column in enumerate(cursor.description):
                if column.type_code == 17:
                    binary.append(i)
                columns.append(column.name)

            for record in cursor:
                rec = list(record)
                for binary_col in binary:
                    rec[binary_col] = binascii.hexlify(record[binary_col])
                tuples.append(rec)

        connection.commit()
        return columns, tuples

    setproctitle.setproctitle("Userspace Collector Process")
    with psycopg.connect("host=localhost port=5432 dbname=benchbase user=wz2", autocommit=True) as connection:
        # Don't allow capturing on the collector instance.
        connection.execute("SET qss_capture_enabled = OFF")
        with connection.cursor() as cursor:
            global collector_pids
            pid = [r for r in cursor.execute("SELECT pg_backend_pid()")][0][0]
            collector_pids.append(pid)

        # Start the phantom.
        query = """
            SELECT phantom_start_worker(ARRAY[{tbls}], {delta})
        """.format(tbls=",".join([f"'{t}'" for t in BENCHDB_TO_TABLES[benchmark]]), delta=slow_time)
        connection.execute(query, prepare=False)

        # Poll on the Collector's output buffer until Collector is shut down.
        while not shutdown.is_set():
            try:
                start = datetime.now()
                knob_values = scrape_settings(connection, KNOBS)
                knob_columns = [k[0] for k in knob_values]
                knob_values = [k[1] for k in knob_values]
                output_columns["pg_settings"] = knob_columns
                output_rows["pg_settings"].append(knob_values)

                for target, query in PG_COLLECTOR_TARGETS.items():
                    columns, tuples = scrape_table(connection, query)
                    output_columns[target] = columns
                    output_rows[target].extend(tuples)

                end = datetime.now()
                delta = (end - start) / timedelta(microseconds=1)
                if delta > (slow_time * 1000000):
                    continue
                time.sleep((slow_time * 1000000 - delta) / 1000000)
            except KeyboardInterrupt:
                logger.info("Userspace Collector caught KeyboardInterrupt.")
            except Exception as e:  # pylint: disable=broad-except
                # TODO(Matt): If postgres shuts down the connection closes and we get an exception for that.
                logger.warning("Userspace Collector caught %s.", e)

        # Make sure that we stall until the phantom worker has shutdown.
        exist = True
        connection.execute("SELECT phantom_stop_worker()", prepare=False)
        while exist:
            exist = bool([r for r in connection.execute("SELECT phantom_worker_exists()", prepare=False)][0][0])

    logger.info("Userspace Collector shut down.")

##################################################################3
# POSTGRES MONITOR
##################################################################3

def monitor_postgres_pid(pid):
    with open("behavior/datagen/probe_templates/postgres_probe.c", "r", encoding="utf-8") as f:
        postgres_probe_c = f.read()

    probes = USDT(pid=pid)
    for probe in ["fork_backend", "reap_backend"]:
        probes.enable_probe(probe=probe, fn_name=probe)

    return BPF(text=postgres_probe_c, usdt_contexts=[probes])

def postmaster_event_cb(postgres_bpf, histograms, processes, collector_flags):
    def postmaster_event(cpu, data, size):
        def create_collector(child_pid, socket_fd=None):
            logger.info("Postmaster forked PID %s, creating its Collector.", child_pid)
            collector_flags[child_pid] = True
            collector_process = mp.Process(
                target=collector, args=(histograms, collector_flags, child_pid, socket_fd)
            )
            collector_process.start()
            processes[child_pid] = collector_process

        def destroy_collector(collector_process, child_pid):
            logger.info("Postmaster reaped PID %s, destroying its Collector.", child_pid)
            collector_flags[child_pid] = False
            collector_process.join()
            del collector_flags[child_pid]
            del processes[child_pid]

        # cpu, size. pylint: disable=unused-argument
        output_event = postgres_bpf["postmaster_events"].event(data)
        event_type = output_event.type_
        child_pid = output_event.pid_
        if event_type == 0:
            fd = output_event.socket_fd_ if event_type == 0 else None
            create_collector(child_pid, fd)
        elif event_type  == 1:
            collector_process = processes.get(child_pid)
            if collector_process:
                destroy_collector(collector_process, child_pid)
        else:
            logger.error("Unknown event type from Postmaster.")
            raise KeyboardInterrupt
    return postmaster_event

##################################################################3
# POSTGRES PER BACKEND COLLECTOR
##################################################################3

def collector(histograms, collector_flags, pid, socket_fd):
    setproctitle.setproctitle(f"{pid} TScout Collector")

    with open("behavior/datagen/probe_templates/probes.c", "r", encoding="utf-8") as markers:
        markers_c = markers.read()

    max_pid = int(open("/proc/sys/kernel/pid_max").read())
    markers_c = markers_c.replace("MAX_PID", str(max_pid))
    markers_c = markers_c.replace("TARGET_PID", str(pid))

    marker_probes = USDT(pid=pid)
    for probe in ["qss_ExecutorStart", "qss_ExecutorEnd", "qss_Block", "qss_Unblock"]:
        marker_probes.enable_probe(probe=probe, fn_name=probe)

    collector_bpf = BPF(text=markers_c, usdt_contexts=[marker_probes])
    collector_bpf.attach_kprobe(event_re="^finish_task_switch$|^finish_task_switch\.isra\.\d$", fn_name="sched_switch")
    collector_bpf.attach_kprobe(event="vfs_read", fn_name="trace_read_entry")
    collector_bpf.attach_kretprobe(event="vfs_read", fn_name="trace_read_return")
    collector_bpf.attach_kprobe(event="vfs_write", fn_name="trace_write_entry")
    collector_bpf.attach_kretprobe(event="vfs_write", fn_name="trace_write_return")

    window_count = 0
    hists = [
        (0, collector_bpf.get_table("dist0")),
        (1, collector_bpf.get_table("dist6")),
        (2, collector_bpf.get_table("dist10")),
    ]
    while collector_flags[pid]:
        try:
            # No perf event, so just sleep...
            if pid not in collector_pids:
                for sl, histogram in hists:
                    hist = { "time": time.time_ns() / 1000, "elapsed_slice": sl, "window_index": window_count, "pid": pid, }
                    for i in range(1, 32):
                        low = (1 << i) >> 1;
                        high = (1 << i) - 1;
                        if low == high:
                            low -= 1
                        key = f"{low}_{high}"

                        try:
                            hist[key] = histogram[i].value
                        except:
                            hist[key] = 0

                    histograms.append(hist)
                window_count += 1

            # Try and get second level histogram resolutions.
            # This way, we can build larger summaries.
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Collector for PID %s caught KeyboardInterrupt.", pid)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Collector for PID %s caught %s.", pid, e)
    logger.info("Collector for PID %s shut down.", pid)


def main(benchmark, outdir, collector_interval, pid, bpf_trace):
    keep_running = True

    # Augment with pgstattuple_approx data.
    #tables = BENCHDB_TO_TABLES[benchmark]
    #for tbl in tables:
    #    PG_COLLECTOR_TARGETS[tbl] = f"SELECT EXTRACT(epoch from NOW())*1000000 as time, * FROM pgstattuple('{tbl}');"
    #for idx in BENCHDB_TO_INDEX[benchmark]:
    #    PG_COLLECTOR_TARGETS[idx] = f"SELECT EXTRACT(epoch from NOW())*1000000 as time, * FROM pgstatindex('{idx}');"

    # Monitor the postgres PID.
    setproctitle.setproctitle("Main Collector Process")
    postgres_bpf = None
    if bpf_trace:
        assert pid is not None, "Postgres PID not specified."
        postgres_bpf = monitor_postgres_pid(pid)

    with mp.Manager() as manager:
        # Create coordination data structures for Collectors and Processors
        collector_flags = manager.dict()
        collector_processes = {}
        histograms = manager.list()

        shutdown = manager.Event()

        # Start the userspace collector.
        pg_scrape_columns = manager.dict()
        pg_scrape_tuples = manager.dict()
        pg_scrape_tuples["pg_settings"] = manager.list()
        for target, _ in PG_COLLECTOR_TARGETS.items():
            pg_scrape_tuples[target] = manager.list()
        pg_collector_process = mp.Process(
            target=pg_collector,
            args=(
                benchmark,
                pg_scrape_tuples,
                pg_scrape_columns,
                collector_interval,
                shutdown,
            ),
        )
        pg_collector_process.start()

        if bpf_trace:
            cb = postmaster_event_cb(postgres_bpf, histograms, collector_processes, collector_flags)
            postgres_bpf["postmaster_events"].open_perf_buffer(callback=cb, lost_cb=lost_something)
            print(f"TScout attached to PID {pid}.")

        while keep_running:
            try:
                if bpf_trace:
                    postgres_bpf.perf_buffer_poll()
                else:
                    time.sleep(collector_interval)
            except KeyboardInterrupt:
                keep_running = False
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Collector caught %s.", e)

        print("Collector shutting down.")

        # Shut down the Collectors so that
        # no more data is generated for the Processors.
        shutdown.set()

        pg_collector_process.join()
        print("Shutdown Userspace Collector")

        if bpf_trace:
            for pid, process in collector_processes.items():
                collector_flags[pid] = False
                process.join()
                logger.info("Joined Collector for PID %s.", pid)
            print("TScout joined all Collectors.")

        PG_COLLECTOR_TARGETS["pg_settings"] = None
        for target in PG_COLLECTOR_TARGETS.keys():
            file_path = f"{outdir}/{target}.csv"
            write_header = not Path(file_path).exists()
            with open(file_path, "a", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(pg_scrape_columns[target])
                writer.writerows(pg_scrape_tuples[target])

        if bpf_trace:
            data = [hist for hist in histograms]
            if len(data) > 0:
                pd.DataFrame(data=data).fillna(0).to_csv(f"{outdir}/histograms.csv", index=False)
        print("Collector wrote out pg collector data.")

        # We're done.
        sys.exit()


class CollectorCLI(cli.Application):
    benchmark = cli.SwitchAttr(
        "--benchmark",
        str,
        mandatory=True,
        help="Benchmark that is being executed.",
    )

    collector_interval = cli.SwitchAttr(
        "--collector_interval",
        int,
        mandatory=False,
        default=60,
        help="TIme between pg collector invocations.",
    )

    outdir = cli.SwitchAttr(
        "--outdir",
        str,
        mandatory=False,
        default=".",
        help="Training data output directory",
    )

    pid = cli.SwitchAttr(
        "--pid",
        int,
        help="Postmaster PID that we're attaching to.",
    )

    bpf_trace = cli.Flag(
        "--bpf-trace",
        help="Whether to use BPF for further tracing.",
    )

    def main(self):
        main(self.benchmark, self.outdir, self.collector_interval, self.pid, self.bpf_trace)


if __name__ == "__main__":
    Collector.run()
