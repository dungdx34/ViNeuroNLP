# ViNeuroNLP
This repository contains the implementation for our paper: Dependency parsing for Vietnamese

### Installation
The code is written in Python 3.6+. Its dependencies are summarized in the file requirements.txt. You can install these dependencies like this:
```shell
pip install -r requirements.txt
```
Note: You must install Java to run the VnCoreNLP library. 

### Models
The models is available at [Google Drive](https://drive.google.com/file/d/1SMwPGOrhPEZecQfCTtiZa7MfTJaKav0G/view?usp=sharing)

Download and unzip the models to the source directory.

### Usage example:
```shell
cd application

# change absolute_path variable in app.py file.

python app.py
```
The input is a sentence. The maximum sentence length is 150 segment words.

The output is formatted with ten columns representing ID, FORM, LEMMA (lowercase of FORM), UPOS, XPOS (same UPOS), FEATS (default: '\_'), HEAD, DEPREL, DEPS (default: '\_'), MISC (default: '\_').

An example is shown below:
```
# sentence : tôi yêu Việt Nam
1       tôi     tôi     PRO     PRO     _       0       root    _       _
2       yêu     yêu     V       V       _       1       compound:vmod   _       _
3       Việt Nam        việt nam        NNP     NNP     _       2       punct   _       _
```

Some code are borrowed from [NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2). Thanks for their work.
