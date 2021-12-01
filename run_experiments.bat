python main.py train_test -c=exp_config/BE_fashion_vgg_2k.json
python main.py train_test -c=exp_config/BE_fashion_vgg_2k_th_05.json
python main.py train_test -c=exp_config/BE_fashion_vgg_th_05.json
python main.py train_test -c=exp_config/MCE10_cifar100_densenet_2k.json
python main.py train_test -c=exp_config/MCE10_cifar100_inception_2k.json
python main.py train_test -c=exp_config/MCE10_cifar100_resnet34_2k.json
python main.py train_test -c=exp_config/MCE10_cifar100_vgg_2k.json
python main.py train_test -c=exp_config/MC_cifar100_densenet.json
python main.py train_test -c=exp_config/MC_cifar100_inception.json
python main.py train_test -c=exp_config/MC_cifar100_resnet34.json
python main.py train_test -c=exp_config/MC_cifar100_vgg.json
python main.py train_test -c=exp_config/MC_fashion_vgg.json
python parse_results.py
echo "Finished experiments execution, please check the results file (exp_config/results.csv)."
