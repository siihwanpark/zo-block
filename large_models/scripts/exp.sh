device=$1

if [ $device -eq 0 ]; then
    bash scripts/run.sh $device regular 1 0 2e-05
elif [ $device -eq 1 ]; then
    bash scripts/run.sh $device MeZO-Adam 10 0 2e-05
elif [ $device -eq 2 ]; then
    bash scripts/run.sh $device MeZO-Adam 10 0 2e-05
elif [ $device -eq 3 ]; then
    bash scripts/run.sh $device MeZO-Adam 10 0 2e-05
elif [ $device -eq 4 ]; then
    bash scripts/run.sh $device MeZO-Adam 1 0 2e-05
    bash scripts/run.sh $device LOZO-Adam 1 0 2e-05
elif [ $device -eq 5 ]; then
    bash scripts/run.sh $device MeZO-Adam 5 0 2e-05
elif [ $device -eq 6 ]; then
    bash scripts/run.sh $device MeZO-Adam 10 0 2e-05
elif [ $device -eq 7 ]; then
    bash scripts/run.sh $device MeZO-Adam 20 0 2e-05
fi