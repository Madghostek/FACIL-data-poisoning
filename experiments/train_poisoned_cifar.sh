cd ..
echo training with $1 buffer size
python3 -u src/main_incremental.py --num-tasks 5 --dataset cifar_10_poisoned --num-exemplars $1
