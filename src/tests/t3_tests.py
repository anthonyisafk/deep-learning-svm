import subprocess

if  __name__ == '__main__':
    main_call = "C:/Users/tonyt/anaconda3/python.exe .\src\smoking.py -s 0 -t 3"
    for r in [i * 0.1 for i in range(-2, 3)]:
        for c in [1, 5, 10, 20, 50]:
            subprocess.call(main_call + f" -r {r} -c {c} -h 0")