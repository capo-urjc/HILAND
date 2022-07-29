#!/bin/bash
cd exp1_5
python main.py train_test -c=exp_config/MC_fashion_vgg.json
python main.py train_test -c=exp_config/BE_fashion_vgg_th_05.json
cd ..
python parse_exp1.py

cd exp1_5
python main.py train_test -c=exp_config/BE_fashion_vgg_2k_th_05.json
cd ..
python parse_exp2.py

cd exp1_5
python main.py train_test -c=exp_config/BE_fashion_vgg_2k.json
cd ..
python parse_exp3.py

cd exp1_5
python main.py train_test -c=exp_config/MCE10_cifar100_densenet_2k.json
python main.py train_test -c=exp_config/MCE10_cifar100_inception_2k.json
python main.py train_test -c=exp_config/MCE10_cifar100_resnet34_2k.json
python main.py train_test -c=exp_config/MCE10_cifar100_vgg_2k.json
cd ..
python parse_exp4.py

cd exp1_5
python main.py train_test -c=exp_config/MC_cifar100_densenet.json
python main.py train_test -c=exp_config/MC_cifar100_inception.json
python main.py train_test -c=exp_config/MC_cifar100_resnet34.json
python main.py train_test -c=exp_config/MC_cifar100_vgg.json
cd ..
python parse_exp5.py

cd exp6_7
python main.py train_test -c=exp_config/MCE100_10_cub200_resnet50_4k.json
cd ..
python parse_exp6.py

cd exp6_7
python main.py train_test -c=exp_config/MCE100_10_cub200_densenet_4k.json
python main.py train_test -c=exp_config/MCE100_10_cub200_inception_4k.json
python main.py train_test -c=exp_config/MCE100_10_cub200_vgg_4k.json
python main.py train_test -c=exp_config/MC_cub200_resnet50.json
python main.py train_test -c=exp_config/MC_cub200_inception.json
python main.py train_test -c=exp_config/MC_cub200_densenet.json
python main.py train_test -c=exp_config/MC_cub200_vgg.json
cd ..
python parse_exp7.py