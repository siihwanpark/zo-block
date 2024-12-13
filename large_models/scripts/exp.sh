device=$1

if [ $device -eq 0 ]; then
    bash scripts/run.sh $device MeZO-SGD 1 1e-6 5e-7 2e-7 1e-7
    bash scripts/run.sh $device MeZO-Adam 1 1e-4 5e-5
elif [ $device -eq 1 ]; then
    bash scripts/run.sh $device LOZO-SGD 1 1e-6 5e-7 2e-7 1e-7
    bash scripts/run.sh $device LOZO-Adam 1 1e-4 5e-5
elif [ $device -eq 2 ]; then
    bash scripts/run.sh $device KFAC-LOZO-SGD 1 1e-4 5e-5 2e-5 1e-5 5e-6
    bash scripts/run.sh $device MeZO-Adam 1 2e-5
elif [ $device -eq 3 ]; then
    bash scripts/run.sh $device KFAC-LOZO-SGD 1 2e-6 1e-6 5e-7 2e-7 1e-7
    bash scripts/run.sh $device LOZO-Adam 1 2e-5
elif [ $device -eq 4 ]; then
    bash scripts/run.sh $device MeZO-Adam 1 1e-5 5e-6 2e-6 1e-6 5e-7 2e-7
elif [ $device -eq 5 ]; then
    bash scripts/run.sh $device LOZO-Adam 1 1e-5 5e-6 2e-6 1e-6 5e-7 2e-7
elif [ $device -eq 6 ]; then
    bash scripts/run.sh $device KFAC-LOZO-Adam 1 1e-4 5e-5 2e-5 1e-5 5e-6
    bash scripts/run.sh $device MeZO-Adam 1 1e-7
elif [ $device -eq 7 ]; then
    bash scripts/run.sh $device KFAC-LOZO-Adam 1 2e-6 1e-6 5e-7 2e-7 1e-7
    bash scripts/run.sh $device LOZO-Adam 1 2e-5
fi