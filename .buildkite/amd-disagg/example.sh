cd ~/work/di-ci/vllm
source /shared_nfs/csrikris/dsr1_0528_env.sh
NODELIST='crsuse2-m2m-v3-[081,082]' bash .buildkite/amd-disagg/run-slurm-disagg-test.sh
