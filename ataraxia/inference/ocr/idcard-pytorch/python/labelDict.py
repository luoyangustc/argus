# -*- coding: utf-8 -*-
from io import open

class genDict():
    def __init__(self, alphabet_path):

        fi = open(alphabet_path, encoding='utf-8')
        alphabet_list = fi.read().strip()
        self.alphabet_list = []
        self.ignor_alphabet_list = ['\ufeff']
        for alpha in alphabet_list:
            if alpha != '\ufeff':
                self.alphabet_list.append(alpha)
        fi.close()

        self.class_num = len(self.alphabet_list)
        self.char_to_label = {}
        self.label_to_char = {}
        for i, char in enumerate(self.alphabet_list):
            if char not in self.ignor_alphabet_list:
                self.char_to_label[char] = i
                self.label_to_char[i] = char

if __name__ == '__main__':
    dict = genDict('/home/sari/PycharmProjects/tableDetector/ForQiNiu_170412/labelsTXT/Number.txt')
    print(dict.alphabet_list)
    print(dict.class_num)
    print(dict.char_to_label)
    print(dict.label_to_char)

