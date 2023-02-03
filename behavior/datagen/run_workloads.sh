#!/bin/bash
set -ex

###
#
# A given workload folder contains the following pieces of information.
#
# 1. A `config.yaml` file that is structured as follows:
#       benchmark: [name of the benchmark to execute]
#       pg_prewarm: True/False [whether to pg_prewarm prior to each benchbase run]
#       pg_analyze: True/False [whether to pg_analyze prior to each benchbase run]
#       pg_configs: [List of files to posgresql.conf files to switch]
#       benchbase_configs: [List of benchbase XML configs for each run]
#
# 2. Relevant postgresql.conf files that should be used for execution
# 3. Relevant BenchBase configuration XMLs that should be used for execution
#
# KNOWN CAVEAT:
#
# 1. `run_workloads.sh` does not handle scale factor shifting between benchbase_configs.
# The benchmark database is loaded with the first benchbase config and then reused
# afterwards. As such, this script does not respect the scale factor of later
# BenchBase configurations (workload distribution & other parameters are respected).
#
# 2. `run_workloads.sh` does not currently support loading multiple databases and having
# benchbase_configs execute differente benchmarks. However, support for this is not difficult
# to add if needed.
#
###

# Various steps may require sudo.
sudo --validate

help() {
    echo "run_workloads.sh [ARGUMENTS]"
    echo ""
    echo "Arguments"
    echo "---------"
    echo "--workloads=[Directory containing all workloads to execute]"
    echo "--output-dir=[Directory to write workload outputs]"
    echo "--pgdata=[Directory of postgres instance to use]"
    echo "--benchbase=[Directory of benchbase installation]"
    echo "--pg_binaries=[Directory containing postgres binaries]"
    exit 1
}

# Parsing --[arg_name]=[arg_value] constructs
arg_parse() {
    for i in "$@"; do
        case $i in
            -workloads=*|--workloads=*)
                WORKLOADS_DIRECTORY="${i#*=}"
                shift # past argument=value
                ;;
            -output-dir=*|--output-dir=*)
                OUTPUT_DIRECTORY="${i#*=}"
                shift # past argument=value
                ;;
            -pgdata=*|--pgdata=*)
                PGDATA_LOCATION="${i#*=}"
                shift # past argument with no value
                ;;
            -benchbase=*|--benchbase=*)
                BENCHBASE_LOCATION="${i#*=}"
                shift # past argument with no value
                ;;
            -pg_binaries=*|--pg_binaries=*)
                PG_BINARIES_LOCATION="${i#*=}"
                PG_CTL_LOCATION="${PG_BINARIES_LOCATION}/pg_ctl"
                PSQL_LOCATION="${PG_BINARIES_LOCATION}/psql"
                PGDUMP_LOCATION="${PG_BINARIES_LOCATION}/pg_dump"
                PGRESTORE_LOCATION="${PG_BINARIES_LOCATION}/pg_restore"
                shift # past argument with no value
                ;;
            -*)
                echo "Unknown option $i"
                help
                ;;
            *)
                ;;
        esac
    done
}

arg_validate() {
    if [ -z ${WORKLOADS_DIRECTORY+x} ] ||
       [ -z ${OUTPUT_DIRECTORY+x} ] ||
       [ -z ${PGDATA_LOCATION+x} ] ||
       [ -z ${BENCHBASE_LOCATION+x} ];
    then
        help
        exit 1
    fi

    if [ ! -d "${WORKLOADS_DIRECTORY}" ];
    then
        echo "Specified workload directory ${WORKLOADS_DIRECTORY} does not exist."
    fi

    if [ ! -d "${BENCHBASE_LOCATION}" ];
    then
        echo "Specified benchbase ${BENCHBASE_LOCATION} does not exist."
    fi

    if [ ! -f "${PG_CTL_LOCATION}" ];
    then
        echo "Specified pg_ctl ${PG_CTL_LOCATION} does not exist."
    fi

    if [ ! -f "${PSQL_LOCATION}" ];
    then
        echo "Specified pg_ctl ${PSQL_LOCATION} does not exist."
    fi
}


# Parse all the input arguments to the bash script.
arg_parse "$@"
# Validate all the input arguments passed to the bash script.
arg_validate
# Record the current timestamp
ts=$(date '+%Y-%m-%d_%H-%M-%S')
echo "Starting workload execution ${ts}"

# Get the absolute file path to the pg_ctl executable
pg_ctl=$(realpath "${PG_CTL_LOCATION}")
psql=$(realpath "${PSQL_LOCATION}")
pg_dump=$(realpath "${PGDUMP_LOCATION}")
pg_restore=$(realpath "${PGRESTORE_LOCATION}")

# Kill any running postgres and/or collector instances.
pkill -i postgres || true
pkill -i collector || true

shopt -s nullglob

output_folder="${OUTPUT_DIRECTORY}/experiment-${ts}/"
workload_directory="${WORKLOADS_DIRECTORY}/"
for workload in "${workload_directory}"/*; do
    echo "Executing ${workload}"

    # Create the output directory for this particular benchmark invocation.
    benchmark_suffix=$(basename "${workload}")
    benchmark_output="${output_folder}/${benchmark_suffix}"
    mkdir -p "${benchmark_output}"

    # Parse the config.yaml file that describes the experiment.
    # The description for the config.yaml and the keys populated are described
    # in behavior/datagen/generate_workloads.py.
    config_yaml=$(realpath "${workload}"/config.yaml)
    (
        eval "$(niet -f eval . "${config_yaml}")"

        # shellcheck disable=2154 # populated by niet
        if [ ${#benchbase_configs[@]} != ${#pg_configs[@]} ];
        then
            echo "Found configuration file ${config_yaml} where configurations are not aligned."
            exit 1
        fi

        if [ ${#benchbase_configs[@]} == 0 ];
        then
            echo "Found configuration file {$config_yaml} containing empty experiment."
            exit 1
        fi

        for i in "${!benchbase_configs[@]}";
        do
            postgresql_path=$(realpath "${pg_configs[$i]}")
            benchbase_config_path=$(realpath "${benchbase_configs[$i]}")

            if [ "$i" -eq 0 ];
            then
                # If we're executing a new experiment, then we want to completely
                # the database instance. This is done by invoking `noisepage_init`.
                doit noisepage_init --config="${postgresql_path}"
                doit benchbase_bootstrap_dbms

                if [ "$restore_db" == 'True' ];
                then
                    ${pg_restore} -j 8 -d benchbase "${restore_db_path}"

                    # These are needed to clean up any sad state from the restore.
                    ${psql} --dbname=benchbase --command="VACUUM;"
                    ${psql} --dbname=benchbase --command="CHECKPOINT;"
                else
                    # Create the database and load the database
                    # shellcheck disable=2154 # populated by niet
                    doit benchbase_run --benchmark="${benchmark}" --config="${benchbase_config_path}" --args="--create=true --load=false --execute=false"

                    if [ -n "$post_create" ];
                    then
                        # Execute post create SQL prior to loading the data.
                        post_create_path=$(realpath "${post_create}")
                        ${psql} --dbname=benchbase -f "${post_create_path}"
                    fi

                    doit benchbase_run --benchmark="${benchmark}" --config="${benchbase_config_path}" --args="--create=false --load=true --execute=false"

                    if [ "$dump_db" == 'True' ];
                    then
                        # Dump the database.
                        dump_file="${benchmark_output}/dump.dir"
                        if [ -n "$dump_db_path" ];
                        then
                            dump_file=$dump_db_path
                        fi
                        ${pg_dump} -j 8 -F d -f "${dump_file}" benchbase
                    fi
                fi

                if [ "$snapshot_data" == 'True' ];
                then
                    doit benchbase_snapshot_benchmark --benchmark="${benchmark}" --output_dir="${benchmark_output}"
                fi

                # Remove existing logfiles, if any exist.
                doit noisepage_stop --data="${PGDATA_LOCATION}"
                rm -rf "${PGDATA_LOCATION}/log/*.csv"
                rm -rf "${PGDATA_LOCATION}/log/*.log"
                rm -rf "${BENCHMARK_LOCATION}/results/*"

                # Then restart the instance.
                ${pg_ctl} start -D "${PGDATA_LOCATION}"

                # Install QSS extension
                doit noisepage_install_extensions --dbname=benchbase

                # shellcheck disable=2154 # populated by niet
                if [ "$pg_analyze" != 'False' ];
                then
                    # If pg_analyze is specified, then run ANALYZE on the benchmark's tables.
                    doit benchbase_pg_analyze_benchmark --benchmark="${benchmark}"
                fi

                # shellcheck disable=2154 # populated by niet
                if [ "$pg_prewarm" != 'False' ];
                then
                    # If pg_prewarm is specified, then invoke pg_prewarm on the benchmark's tables.
                    doit benchbase_warm_benchmark --benchmark="${benchmark}"
                fi
            elif [ "${continuous}" != 'True' ];
            then
                # Under continuous execution, we don't swap the configuration.
                # Swapping the configuration forces a restart at present.
                doit noisepage_swap_config --config="${postgresql_path}"

                # shellcheck disable=2154 # populated by niet
                if [ "$pg_analyze" != 'False' ];
                then
                    # If pg_analyze is specified, then run ANALYZE on the benchmark's tables.
                    doit benchbase_pg_analyze_benchmark --benchmark="${benchmark}"
                fi

                # shellcheck disable=2154 # populated by niet
                if [ "$pg_prewarm" != 'False' ];
                then
                    # If pg_prewarm is specified, then invoke pg_prewarm on the benchmark's tables.
                    doit benchbase_warm_benchmark --benchmark="${benchmark}"
                fi
            fi

            postmaster_pid=$(pidof postgres | xargs -n1 | sort | head -n1)
            if [ ! -z "$taskset_postgres" ] && [ "$taskset_postgres" != 'None' ];
            then
                taskset -pc $taskset_postgres $postmaster_pid
            fi

            if [ "$snapshot_metadata" == 'True' ];
            then
                ${psql} --dbname=benchbase --csv --command="SELECT EXTRACT(epoch from NOW()) as time, * FROM pg_stats s JOIN information_schema.columns c ON s.tablename=c.table_name AND s.attname=c.column_name WHERE s.schemaname = 'public';" > "${benchmark_output}/pg_stats.csv.${i}"
                ${psql} --dbname=benchbase --csv --command="SELECT EXTRACT(epoch from NOW()) as time, * FROM pg_class t JOIN pg_namespace n ON n.oid = t.relnamespace WHERE n.nspname = 'public';" > "${benchmark_output}/pg_class.csv.${i}"
                ${psql} --dbname=benchbase --csv --command="SELECT EXTRACT(epoch from NOW()) as time, * FROM pg_attribute;" > "${benchmark_output}/pg_attribute.csv.${i}"
                ${psql} --dbname=benchbase --csv --command="SELECT EXTRACT(epoch from NOW()) as time, * FROM pg_index;" > "${benchmark_output}/pg_index.csv.${i}"
            fi

            if [ "$enable_collector" != 'False' ];
            then
                # Initialize collector. We currently don't have a means by which to check whether
                # collector has successfully attached to the instance. As such, we (wait) 10 seconds.
                doit collector_init --benchmark="${benchmark}" --output_dir="${benchmark_output}" --wait_time=60 --collector_interval=$collector_interval --pid=$postmaster_pid
            fi

            if [ ! -z "$pre_execute" ];
            then
                preexecute_path=$(realpath "${pre_execute[$i]}")
                ${psql} --dbname=benchbase -f "${preexecute_path}"
            fi

            # Execute the benchmark
            doit benchbase_run --benchmark="${benchmark}" --config="${benchbase_config_path}" --args="--execute=true" --taskset_benchbase="$taskset_benchbase"

            if [ "$enable_collector" != 'False' ];
            then
                # Shutdown collector.
                doit collector_shutdown --output_dir="${benchmark_output}"
            fi

            # Move the relevant stats out to the benchmark output.
            mkdir "${benchmark_output}/stats.${i}"
            mkdir "${benchmark_output}/phantom.${i}"
            mv ${PGDATA_LOCATION}/pg_stat/*.csv "${benchmark_output}/phantom.${i}/"
            mv ${benchmark_output}/phantom.${i}/pg_qss_*.csv "${benchmark_output}/stats.${i}/"

            if [ ! -z "$post_execute" ];
            then
                postexecute_path=$(realpath "${post_execute[$i]}")
                ${psql} --dbname=benchbase -f "${postexecute_path}"
            fi

            # Similarly, we move the postgres log file to the experiment output directory if it
            # exists. The log file is also suffixed by this benchmark index.
            log=${PGDATA_LOCATION}/log
            if [ "${continuous}" != 'True' ];
            then
                doit noisepage_stop --data="${PGDATA_LOCATION}"
            fi

            if [ -d "${log}" ] && [ "${continuous}" != 'True' ];
            then
                mv "${PGDATA_LOCATION}/log" "${benchmark_output}/log.${i}"
            elif [ -d "${log}" ];
            then
                # Make and move the log files.
                mkdir "${benchmark_output}/log.${i}"
                mv ${PGDATA_LOCATION}/log/* "${benchmark_output}/log.${i}/"
                ${psql} --dbname=benchbase --csv --command="SELECT * FROM pg_rotate_logfile()"
            fi

            # Similarly, we move the corresponding benchmark's execution log from BenchBase to the
            # experiment output directory, with the results folder suffixed by the benchmark index.
            mv "${BENCHBASE_LOCATION}/results" "${benchmark_output}/results.${i}"
        done

        if [ "${continuous}" == 'True' ];
        then
            doit noisepage_stop --data="${PGDATA_LOCATION}"
            mv "${PGDATA_LOCATION}/log" "${benchmark_output}/log"
        fi
    )

    echo "Executed ${workload}"
done

ts=$(date '+%Y-%m-%d_%H-%M-%S')
echo "Finished workload execution at ${ts}"
