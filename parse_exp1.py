from pathlib import Path

from parse_results import main

if __name__ == '__main__':
    smart_override = False

    search_path = Path("exp1_5/exp_out/")

    run_names = ["MC_fashion_vgg", "BE_fashion_vgg_th_05"]

    exp_name = 'exp_out/exp1.csv'

    main(search_path, run_names, exp_name, smart_override)
