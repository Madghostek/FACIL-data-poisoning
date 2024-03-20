cd ..
echo training with $1 buffer size
python3 -u src/main_incremental.py --dataset cifar_poisoned --num-tasks 5 --num-exemplars $1
