cd ..
echo training with $1 epoch $2 lamb
python3 -u src/main_incremental.py --approach lwf --seed 1234 --dataset cifar10 --num-tasks 5 --nepochs $1 --lamb $2 
