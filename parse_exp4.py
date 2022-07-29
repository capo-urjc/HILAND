from pathlib import Path

from parse_results import main

if __name__ == '__main__':
    smart_override = True

    search_path = Path("exp1_5/exp_out")

    run_names = ["MCE10_cifar100_densenet_2k",
                 "MCE10_cifar100_inception_2k",
                 "MCE10_cifar100_resnet34_2k",
                 "MCE10_cifar100_vgg_2k"]

    exp_name = 'exp_out/exp4.csv'

    main(search_path, run_names, exp_name, smart_override)
