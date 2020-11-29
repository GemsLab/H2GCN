#!/bin/bash
FAIL=0
trap 'echo Exiting...; kill $(jobs -pr)' SIGINT SIGTERM

set -x

python run_hgcn_experiments.py exec clear_workspace $@ &
python run_gcn_experiments.py exec clear_workspace $@ &
python run_mixhop_experiments.py exec clear_workspace $@ &
python run_gat_experiments.py exec clear_workspace $@ &
python run_graphsage_experiments.py exec clear_workspace $@ &

set +x

for job in `jobs -p`
do
    wait $job || let "FAIL+=1"
done

echo "$FAIL jobs failed"


