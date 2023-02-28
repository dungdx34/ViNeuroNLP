# -*- coding: utf-8 -*-

__author__ = 'Dung Doan'

import os
import sys

sys.path.append(".")
sys.path.append("..")

from application.vidp import ViDP

if __name__ == '__main__':

    absolute_path = "/absolute/path/to/ViNeuroNLP"
    vi_parser = ViDP(absolute_path)

    sentence = "tôi yêu Việt Nam"
    print(sentence)
    result = vi_parser.parse(sentence=sentence)
    print(result)

    fin = open(os.path.join(absolute_path, "demo.txt"), 'r', encoding='utf-8')
    for line in fin:
        line = line.strip()
        print(line)
        result = vi_parser.parse(sentence=line)
        print(result)
