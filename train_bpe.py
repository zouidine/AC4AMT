from AC4AMT.preprocess import Preprocessing
import youtokentome as yttm
import codecs

data_path = "./AC4AMT/data/"

ar_lang = "ar"
en_lang = "en"

ar_preprocessor = Preprocessing(ar_lang)
en_preprocessor = Preprocessing(en_lang)

train_ar = open('{}train_{}.txt'.format(data_path, ar_lang), 
                encoding='utf-8').read().strip().split('\n')
train_en = open('{}train_{}.txt'.format(data_path, en_lang),
                encoding='utf-8').read().strip().split('\n')

train_ar = ar_preprocessor.clean(train_ar)
train_en = en_preprocessor.clean(train_en)

with codecs.open("train_ar.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_ar))
with codecs.open("train_en.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_en))

# Training model
yttm.BPE.train(data='train_en.txt', 
               vocab_size=10000, model="./models/bpe_en.model")
yttm.BPE.train(data='train_ar.txt', 
               vocab_size=10000, model="./models/bpe_ar.model")
