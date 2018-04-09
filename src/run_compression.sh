#!/bin/bash
##############################################
# ./work.sh <gpu_id> <lamb> <smax> <fn_base_out>
##############################################

# Compression experiment: ./work.sh <gpu_id> <c> <smax> <path_log>_db.pkz
CUDA_VISIBLE_DEVICES=$1 python run.py --experiment=mixture --approach=hat-test --parameter=$2,$3,$4_facescrub.pkz,1 --seed=0
CUDA_VISIBLE_DEVICES=$1 python run.py --experiment=mixture --approach=hat-test --parameter=$2,$3,$4_notmnist.pkz,1 --seed=1
CUDA_VISIBLE_DEVICES=$1 python run.py --experiment=mixture --approach=hat-test --parameter=$2,$3,$4_fashion-mnist.pkz,1 --seed=2
CUDA_VISIBLE_DEVICES=$1 python run.py --experiment=mixture --approach=hat-test --parameter=$2,$3,$4_traffic-signs.pkz,1 --seed=3
CUDA_VISIBLE_DEVICES=$1 python run.py --experiment=mixture --approach=hat-test --parameter=$2,$3,$4_cifar10.pkz,1 --seed=6
CUDA_VISIBLE_DEVICES=$1 python run.py --experiment=mixture --approach=hat-test --parameter=$2,$3,$4_mnist.pkz,1 --seed=10
CUDA_VISIBLE_DEVICES=$1 python run.py --experiment=mixture --approach=hat-test --parameter=$2,$3,$4_cifar100.pkz,1 --seed=13
CUDA_VISIBLE_DEVICES=$1 python run.py --experiment=mixture --approach=hat-test --parameter=$2,$3,$4_svhn.pkz,1 --seed=25
