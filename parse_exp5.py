from pathlib import Path

from parse_results import main

if __name__ == '__main__':
    smart_override = False

    search_path = Path("exp1_5/exp_out")

    run_names = ["MC_cifar100_densenet",
                 "MC_cifar100_inception",
                 "MC_cifar100_resnet34",
                 "MC_cifar100_vgg", ]

    exp_name = 'exp_out/exp5.csv'

    main(search_path, run_names, exp_name, smart_override)
