cd ..
echo training with $1 epoch $2 lamb $3 seed
python3 -u src/main_incremental.py --approach lwf --seed $3 --dataset cifar10 --num-tasks 5 --nepochs $1 --lamb $2