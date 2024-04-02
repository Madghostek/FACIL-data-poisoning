cd ..
echo training with $1 buffer size and $2 epochs
python3 -u src/main_incremental.py --dataset cifar_10_poisoned --num-tasks 5 --num-exemplars $1 --nepochs $2
