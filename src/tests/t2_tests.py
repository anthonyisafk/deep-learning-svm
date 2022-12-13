import subprocess

if __name__ == '__main__':
    main_call = "C:/Users/tonyt/anaconda3/python.exe .\src\smoking.py -s 0 -t 2"
    # for g in [0.03, 0.04, 0.05]:
    for w0 in [1, 2, 5, 10, 20]:
        for w1 in [1, 2, 5, 10, 20]:
            subprocess.call(main_call + f" -g 0.01 -w0 {w0} -w1 {w1} -h 0")