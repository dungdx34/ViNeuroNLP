__author__ = 'max'

from neuronlp2.io.instance import DependencyInstance, NERInstance, DependencyBertInstance
from neuronlp2.io.instance import Sentence
from neuronlp2.io.common import ROOT, ROOT_POS, ROOT_CHAR, ROOT_TYPE, END, END_POS, END_CHAR, END_TYPE
from neuronlp2.io.common import DIGIT_RE, MAX_CHAR_LENGTH
import torch
from neuronlp2.io.utils import get_main_deplabel

class CoNLLXReader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False, use_test=False):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            lines.append(line.split('\t'))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        types = []
        type_ids = []
        heads = []

        if symbolic_root:
            words.append(ROOT)
            word_ids.append(self.__word_alphabet.get_index(ROOT))
            char_seqs.append([ROOT_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(ROOT_CHAR), ])
            postags.append(ROOT_POS)
            pos_ids.append(self.__pos_alphabet.get_index(ROOT_POS))
            types.append(ROOT_TYPE)
            type_ids.append(self.__type_alphabet.get_index(ROOT_TYPE))
            heads.append(0)

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            # word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
            word = tokens[1]

            pos = tokens[4]

            if not use_test:
                head = int(tokens[6])
                # type = get_main_deplabel(tokens[7])
                type = tokens[7]
            else:
                head = 0
                type = 'nmod'

            # if pos == 'CH':
            #     pos = 'PUNCT'
            # elif pos == 'L':
            #     pos = 'DET'
            # elif pos == 'A':
            #     pos = 'ADJ'
            # elif pos == 'R':
            #     pos = 'ADJ'
            # elif pos == 'Np':
            #     pos = 'NNP'
            # elif pos == 'M':
            #     pos = 'NUM'
            # elif pos == 'E':
            #     pos = 'PRE'
            # elif pos == 'P':
            #     pos = 'PRO'
            # elif pos == 'Cc':
            #     pos = 'CC'
            # elif pos == 'T':
            #     pos = 'AUX'
            # elif pos == 'Y':
            #     pos = 'NNP'
            # elif pos == 'Cb':
            #     pos = 'CC'
            # elif pos == 'Eb':
            #     pos = 'FW'
            # elif pos == 'Ni':
            #     pos = 'Ny'
            # elif pos == 'B':
            #     pos = 'NNP'
            #
            # if pos == 'L':
            #     pos = 'DET'
            # elif pos == 'Aux':
            #     pos = 'AUX'
            # elif pos == 'NN':
            #     pos = 'N'
            #
            # if pos == '_':
            #     pos = 'MW'

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            types.append(type)
            type_ids.append(self.__type_alphabet.get_index(type))

            heads.append(head)

        if symbolic_end:
            words.append(END)
            word_ids.append(self.__word_alphabet.get_index(END))
            char_seqs.append([END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])
            postags.append(END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(END_POS))
            types.append(END_TYPE)
            type_ids.append(self.__type_alphabet.get_index(END_TYPE))
            heads.append(0)

        return DependencyInstance(Sentence(words, word_ids, char_seqs, char_id_seqs), postags, pos_ids, heads, types, type_ids)

class CoNLLXReaderTransform(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, feature_bert_path, use_bert=False):
        self.__source_file = open(file_path, 'r', encoding='utf-8')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet
        self.count = 0

        if use_bert:
            saved = torch.load(feature_bert_path)
            self.features_bert = []
            for i in range(len(saved)):
                self.features_bert.append(saved[i])

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False, use_bert=False, use_test=False):
        lines = []
        line = self.__source_file.readline()

        while line is not None and len(line.strip()) > 0:
            line = line.strip()
            # line = line.decode('utf-8')
            if '# sent_id' in line or '# text' in line or '# newdoc' in line or '# source' in line or '# origin' in line or '# orig' in line:
                line = self.__source_file.readline()
                continue
            lines.append(line.split('\t'))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        types = []
        type_ids = []
        heads = []

        if symbolic_root:
            words.append(ROOT)
            word_ids.append(self.__word_alphabet.get_index(ROOT))
            char_seqs.append([ROOT_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(ROOT_CHAR), ])
            postags.append(ROOT_POS)
            pos_ids.append(self.__pos_alphabet.get_index(ROOT_POS))
            types.append(ROOT_TYPE)
            type_ids.append(self.__type_alphabet.get_index(ROOT_TYPE))
            heads.append(0)

        for tokens in lines:
            chars = []
            char_ids = []

            if '-' in tokens[0] or '.' in tokens[0]:
                continue

            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            # word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]

            word = tokens[1]
            # word = word.lower()

            pos = tokens[4]

            if not use_test:
                head = int(tokens[6])
                # type = get_main_deplabel(tokens[7])
                type = tokens[7]
            else:
                head = 0
                type = 'nmod'

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            # if pos == 'CH':
            #     pos = 'PUNCT'
            # elif pos == 'L':
            #     pos = 'DET'
            # elif pos == 'A':
            #     pos = 'ADJ'
            # elif pos == 'R':
            #     pos = 'ADJ'
            # elif pos == 'Np':
            #     pos = 'NNP'
            # elif pos == 'M':
            #     pos = 'NUM'
            # elif pos == 'E':
            #     pos = 'PRE'
            # elif pos == 'P':
            #     pos = 'PRO'
            # elif pos == 'Cc':
            #     pos = 'CC'
            # elif pos == 'T':
            #     pos = 'AUX'
            # elif pos == 'Y':
            #     pos = 'NNP'
            # elif pos == 'Cb':
            #     pos = 'CC'
            # elif pos == 'Eb':
            #     pos = 'FW'
            # elif pos == 'Ni':
            #     pos = 'Ny'
            # elif pos == 'B':
            #     pos = 'NNP'
            #
            # if pos == 'L':
            #     pos = 'DET'
            # elif pos == 'Aux':
            #     pos = 'AUX'
            # elif pos == 'NN':
            #     pos = 'N'
            #
            # if pos == '_':
            #     pos = 'MW'

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            types.append(type)
            type_ids.append(self.__type_alphabet.get_index(type))

            heads.append(head)

        if symbolic_end:
            words.append(END)
            word_ids.append(self.__word_alphabet.get_index(END))
            char_seqs.append([END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])
            postags.append(END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(END_POS))
            types.append(END_TYPE)
            type_ids.append(self.__type_alphabet.get_index(END_TYPE))
            heads.append(0)

        if use_bert:
            bert_ids = self.features_bert[self.count]
            self.count += 1
            return DependencyBertInstance(Sentence(words, word_ids, char_seqs, char_id_seqs), postags, pos_ids, heads,
                                          types,
                                          type_ids, bert_ids)
        else:
            return DependencyInstance(Sentence(words, word_ids, char_seqs, char_id_seqs), postags, pos_ids, heads,
                                      types, type_ids)

class CoNLL03Reader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__chunk_alphabet = chunk_alphabet
        self.__ner_alphabet = ner_alphabet

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            lines.append(line.split(' '))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        chunk_tags = []
        chunk_ids = []
        ner_tags = []
        ner_ids = []

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
            pos = tokens[2]
            chunk = tokens[3]
            ner = tokens[4]

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            chunk_tags.append(chunk)
            chunk_ids.append(self.__chunk_alphabet.get_index(chunk))

            ner_tags.append(ner)
            ner_ids.append(self.__ner_alphabet.get_index(ner))

        return NERInstance(Sentence(words, word_ids, char_seqs, char_id_seqs),
                           postags, pos_ids, chunk_tags, chunk_ids, ner_tags, ner_ids)
