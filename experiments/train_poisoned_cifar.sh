cd ..
echo training with $1 buffer size and $2 epochs
python3 -u src/main_incremental.py --approach joint --dataset cifar_10_poisoned --num-tasks 5 --nepochs $2 --num-exemplars $1 
