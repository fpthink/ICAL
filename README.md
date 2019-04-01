# Inter-Class Angular Loss for Convolutional Neural Networks

Pytorch 0.3


Run:
CUDA_VISIBLE_DEVICES=0 python cifar10.py -a preresnet --depth 110 --epochs 200 --schedule 100 150 --lam 5.0 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10_ICAL/exp

CUDA_VISIBLE_DEVICES=0 python noaug_cifar10.py -a preresnet --depth 110 --epochs 200 --schedule 100 150 --lam 10.0 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10_ICAL/noaug_exp


