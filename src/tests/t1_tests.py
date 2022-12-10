import sys
import subprocess

if __name__ == '__main__':
    # main_call = "C:/Users/tonyt/anaconda3/python.exe .\src\smoking.py -s 0 -t 1"
    # for d in [2, 3, 4, 5, 6, 7, 8, 9]:
    #     for r in [i * 0.1 for i in range(-5, 5)]:
    #         for c in [10, 100, 1000, 10000, 100000]:
    #             for h in [0, 1]:
    #                 subprocess.call(main_call + f" -d {d} -r {r} -c {c} -h {h}")

    main_call = "C:/Users/tonyt/anaconda3/python.exe .\src\smoking.py -s 0 -t 1"
    for d in [2, 3, 4, 5, 6, 7, 8, 9]:
        for r in [i * 0.1 for i in range(-1, 2)]:
            for c in [1, 5, 10, 20, 50]:
                # for h in [0, 1]:
                subprocess.call(main_call + f" -d {d} -r {r} -c {c} -h 0")