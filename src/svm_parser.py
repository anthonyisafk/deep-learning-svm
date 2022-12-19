import argparse
import logging


def SVMParser(default_g):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-s', help='SVM type', type=int, default=0)
    parser.add_argument('-t', help='kernel type', type=int, default=0)
    parser.add_argument('-d', help='polynomial degree', type=int, default=1)
    parser.add_argument('-g', help='gamma', type=float, default=default_g)
    parser.add_argument('-r', help='offset', type=float, default=0)
    parser.add_argument('-c', help='cost', type=int, default=1)
    parser.add_argument('-w0', help="class 0 weight", type=float, default=1.0)
    parser.add_argument('-w1', help="class 1 weight", type=float, default=1.0)
    parser.add_argument('-h', help='use of shrinking', type=int, default=0)
    return parser


def get_params_csv(args):
    kernel_type  = args.t
    if kernel_type == 0:
        return f"{args.s},{args.t},{args.c},{args.w0},{args.w1},{args.h}"
    if kernel_type == 1:
        return f"{args.s},{args.t},{args.d},{args.g},{args.r},{args.c},{args.h}"
    if kernel_type == 2:
        return f"{args.s},{args.t},{args.g},{args.c},{args.w0},{args.w1},{args.h}"
    if kernel_type == 3:
        return f"{args.s},{args.t},{args.g},{args.r},{args.c},{args.h}"


def get_params_str(parser:argparse.ArgumentParser):
    params = ""
    args = parser.parse_args().__dict__
    for i in args:
        params += f"-{i} {args[i]} "
    return params