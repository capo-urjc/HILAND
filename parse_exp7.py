from pathlib import Path

from parse_results import main

if __name__ == '__main__':
    smart_override = True

    search_path = Path("exp6_7/exp_out")

    run_names = ["MCE100_10_cub200_densenet_4k",
                 "MCE100_10_cub200_inception_4k",
                 "MCE100_10_cub200_vgg_4k",
                 "MC_cub200_densenet",
                 "MC_cub200_inception",
                 "MC_cub200_resnet50",
                 "MC_cub200_vgg"]

    exp_name = 'exp_out/exp7.csv'

    main(search_path, run_names, exp_name, smart_override, [100, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
