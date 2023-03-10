__author__ = 'Dung Doan'

import json
import torch
from neuronlp2.io import get_logger
from neuronlp2.io import conllx_data, iterate_data
from neuronlp2.io import CoNLLXWriter
from neuronlp2.models import DeepBiAffineTransformNew
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from transformers import AutoTokenizer
from transformers import AutoModel
import os, re
import py_vncorenlp
from application.utils import DPProcessor, InputFeatures
import phonlp

ID = "id"
FORM = "form"
LEMMA = 'lemma'
UPOS = 'upos'
XPOS = 'xpos'
FEATS = 'feats'
HEAD = 'head'
DEPREL = 'deprel'
DEPS = 'deps'
MISC = 'misc'

class ViDP():
    def __init__(self, directory, tmp_dir="tmp"):
        vncore_path = os.path.join(directory, "VnCoreNLP")
        phonlp_path = os.path.join(directory, "models")
        model_path = os.path.join(directory, "models/deepbiaf_bert")
        phobert_path = os.path.join(directory, "models/phobert-base")
        self.tmp_path = os.path.join(directory, tmp_dir)

        self.annotator = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncore_path)

        self.pos_anotator = phonlp.load(phonlp_path)

        self.logger = get_logger("DP_API")



        alphabet_path = os.path.join(model_path, 'alphabets/')
        model_name = os.path.join(model_path, 'network.pt')
        word_alphabet, char_alphabet, pos_alphabet, \
            type_alphabet, _ = conllx_data.create_alphabets(alphabet_path, None,
                                                                          data_paths=[None, None],
                                                                          max_vocabulary_size=50000,
                                                                          embedd_dict=None)

        num_words = word_alphabet.size()
        num_chars = char_alphabet.size()
        num_pos = pos_alphabet.size()
        num_types = type_alphabet.size()

        self.logger.info("Word Alphabet Size: %d" % num_words)
        self.logger.info("Character Alphabet Size: %d" % num_chars)
        self.logger.info("POS Alphabet Size: %d" % num_pos)
        self.logger.info("Type Alphabet Size: %d" % num_types)

        self.word_alphabet = word_alphabet
        self.char_alphabet  = char_alphabet
        self.pos_alphabet = pos_alphabet
        self.type_alphabet = type_alphabet

        def load_model_arguments_from_json():
            arguments = json.load(open(arg_path, 'r'))
            return arguments

        arg_path = os.path.join(model_path, 'config.json')

        hyps = load_model_arguments_from_json()
        model_type = hyps['model']
        word_dim = hyps['word_dim']
        char_dim = hyps['char_dim']
        use_pos = hyps['pos']
        pos_dim = hyps['pos_dim']
        mode = hyps['rnn_mode']
        hidden_size = hyps['hidden_size']
        arc_space = hyps['arc_space']
        type_space = hyps['type_space']
        p_in = hyps['p_in']
        p_out = hyps['p_out']
        p_rnn = hyps['p_rnn']
        activation = hyps['activation']
        num_layers = hyps['num_layers']

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            use_gpu = True
        else:
            self.device = torch.device("cpu")
            use_gpu = False

        self.network = DeepBiAffineTransformNew(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                                                       mode, hidden_size, num_layers, num_types, arc_space, type_space,
                                                       p_in=p_in, p_out=p_out, p_rnn=p_rnn,
                                                       pos=use_pos, activation=activation,
                                                       use_con_rnn=True,
                                                       use_gpu=use_gpu, no_word=False,
                                                       use_bert=True)

        self.tokenizer = AutoTokenizer.from_pretrained(phobert_path)
        self.model_bert = AutoModel.from_pretrained(phobert_path)

        self.network.to(self.device)
        self.model_bert.to(self.device)

        self.network.load_state_dict(torch.load(model_name, map_location=self.device))

    def word_segmentation(self, text):
        try:
            text = re.sub("\s+", " ", text)
            postags = []
            new_text = self.annotator.word_segment(text)[0]
            words = new_text.split(" ")

            if len(words) > 150:
                words = words[0:150]
                new_text = " ".join(words)

            new_sentences = self.pos_anotator.annotate(new_text)
            new_postags = new_sentences[1][0]
            for pos in new_postags:
                postags.append(pos[0])

            return words, postags
        except Exception as err:
            print(err)
        return [], []

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        features = []
        unk_token = tokenizer.unk_token
        cls_token_segment_id = 0
        sequence_a_segment_id = 0
        pad_token_id = tokenizer.pad_token_id
        pad_token_segment_id = 0
        mask_padding_with_zero = True
        start_token = 0
        end_token = 2

        for (ex_index, example) in enumerate(examples):
            words = example.text_a
            tokens = []

            for i, word in enumerate(words):
                word = word.replace(" ", "_")

                word_tokens = tokenizer.encode(word)
                word_tokens = word_tokens[1:-1]
                if len(word_tokens) > 1:
                    word_tokens = [word_tokens[0]]

                if not word_tokens:
                    word_tokens = [unk_token]

                tokens.extend(word_tokens)

            special_tokens_count = 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[:(max_seq_length - special_tokens_count)]

            tokens += [end_token]
            token_type_ids = [sequence_a_segment_id] * len(tokens)

            tokens = [start_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids

            input_ids = tokens
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
            assert len(input_mask) == max_seq_length, "Error with attention mask length {} vs {}".format(len(input_mask), max_seq_length)
            assert len(token_type_ids) == max_seq_length, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                      max_seq_length)
            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids))

        return features

    def parse(self, sentence):
        result_segment = ''
        words, postags = self.word_segmentation(sentence)

        for index, (word, pos) in enumerate(zip(words, postags)):
            word = word.replace("_", " ")
            if pos == 'CH':
                pos = 'PUNCT'
            elif pos == 'L':
                pos = 'DET'
            elif pos == 'A':
                pos = 'ADJ'
            elif pos == 'R':
                pos = 'ADV'
            elif pos == 'Np':
                pos = 'NNP'
            elif pos == 'M':
                pos = 'NUM'
            elif pos == 'E':
                pos = 'PRE'
            elif pos == 'P':
                pos = 'PRO'
            elif pos == 'Cc':
                pos = 'CC'
            elif pos == 'T':
                pos = 'PART'
            elif pos == 'PART':
                pos = 'PART'
            elif pos == 'Y':
                pos = 'NNP'
            elif pos == 'Cb':
                pos = 'CC'
            elif pos == 'Eb':
                pos = 'FW'
            elif pos == 'Ni':
                pos = 'Ny'
            elif pos == 'B':
                pos = 'NNP'
            elif pos == 'L':
                pos = 'DET'
            elif pos == 'Aux':
                pos = 'AUX'
            elif pos == 'NN':
                pos = 'N'

            result_segment += '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(index+1, word, word.lower(), pos, pos, '_', '_', '_', '_', '_') + '\n'

        result_segment = result_segment.strip()

        test_folder = self.tmp_path
        if not os.path.exists(test_folder):
            os.mkdir(test_folder)
        else:
            for file in os.listdir(test_folder):
                os.remove(os.path.join(test_folder, file))

        output_path = os.path.join(test_folder, 'test.conll')
        fout = open(output_path, 'w', encoding='utf-8')
        fout.write(result_segment + '\n')
        fout.close()

        processor = DPProcessor()

        test_path = os.path.join(self.tmp_path, 'test.conll')
        feature_bert_path = os.path.join(self.tmp_path, 'phobert_features.pth')
        train_examples = processor.get_train_examples(test_path)
        all_lengths = []
        for t in train_examples:
            all_lengths.append(len(t.text_a))
        max_seq_len = max(all_lengths) + 1

        train_features = self.convert_examples_to_features(train_examples, max_seq_len, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=5)

        self.model_bert.eval()
        to_save = {}

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, token_type_ids = batch

            with torch.no_grad():
                all_encoder_layers = self.model_bert(input_ids, attention_mask=input_mask,
                                                token_type_ids=token_type_ids)

            output_ = all_encoder_layers[0]

            for j in range(len(input_ids)):
                sent_id = j + step * 5
                layer_output = output_[j, :input_mask[j].to('cpu').sum()]
                to_save[sent_id] = layer_output.detach().cpu().numpy()

        torch.save(to_save, feature_bert_path)

        data_test = conllx_data.read_data_transform(test_path, self.word_alphabet, self.char_alphabet, self.pos_alphabet,
                                                    self.type_alphabet, feature_bert_path,
                                                    use_bert=True,
                                                    symbolic_root=True, use_test=True)

        pred_writer = CoNLLXWriter(self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.type_alphabet)

        self.network.eval()

        out_filename = os.path.join(self.tmp_path, 'test')
        pred_writer.start(out_filename + '_pred.conll')

        with torch.no_grad():
            self.network.eval()

            for data in iterate_data(data_test, batch_size=1):
                words = data['WORD'].to(self.device)
                chars = data['CHAR'].to(self.device)
                postags = data['POS'].to(self.device)
                lengths = data['LENGTH'].numpy()
                berts = data['BERT'].to(self.device)

                masks = data['MASK'].to(self.device)
                heads_pred, types_pred = self.network.decode(words, chars, postags, berts, mask=masks,
                                                        leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)

                words = words.cpu().numpy()
                postags = postags.cpu().numpy()
                pred_writer.write(words, postags, heads_pred, types_pred, lengths, symbolic_root=True)

        pred_writer.close()

        results = {
            ID: [],
            FORM: [],
            LEMMA: [],
            UPOS: [],
            XPOS: [],
            FEATS: [],
            HEAD: [],
            DEPREL: [],
            DEPS :[],
            MISC: []
        }

        sents_gold = result_segment.split('\n')
        test_path = os.path.join(self.tmp_path, 'test_pred.conll')
        lines = open(test_path, 'r', encoding='utf-8').readlines()
        for i, line in enumerate(lines):
            if line.strip() != '':
                sent = sents_gold[i]
                words_gold = sent.split('\t')
                word = words_gold[1]

                line = line.strip()
                words = line.split('\t')

                results[ID].append(words[0] )
                results[FORM].append(word)
                results[LEMMA].append(word.lower())
                results[UPOS].append(words[4])
                results[XPOS].append(words[4])
                results[FEATS].append("_")

                results[HEAD].append(words[6])
                results[DEPREL].append(words[7])

                results[DEPS].append("_")
                results[MISC].append("_")

        out_sent = []
        for index in range(len(results[ID])):
            out_sent.append('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                results[ID][index], results[FORM][index], results[LEMMA][index],
                results[UPOS][index], results[XPOS][index], results[FEATS][index], results[HEAD][index],
                results[DEPREL][index], results[DEPS][index], results[MISC][index]
            ))

        out_doc = '\n'.join(out_sent)

        return out_doc


