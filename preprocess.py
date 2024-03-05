import torch
import re

class Preprocessing():

    def __init__(self, lang):
        self.lang = lang

    # English preprocessing
    def en_clean(self, sentence):
        return re.sub(r"[^a-zA-Z0-9.?!' ]+", "", sentence).lower()

    # Arabic preprocessing
    def ar_clean(self, sentence):
        #remove punctuations
        arabic_punctuations = '''`÷×؛<>_()*&^%][،/:"'{}~¦+|”…“–ـ\#=-,٬@—‘♫;٪อรอย$♪'''
        translator = str.maketrans('', '', arabic_punctuations)
        sentence = sentence.translate(translator)
        #hindi numbers to arabic numbers
        hindi_nums = "٠١٢٣٤٥٦٧٨٩"
        arabic_nums = "0123456789"
        hindi_to_arabic_map = str.maketrans(hindi_nums, arabic_nums)
        sentence = sentence.translate(hindi_to_arabic_map)
        #remove elongations
        sentence = re.sub(r'(.)\1+', r'\1', sentence)
        #remove diacritics
        arabic_diacritics = re.compile("""
                                ّ    | # Tashdid
                                َ    | # Fatha
                                ً    | # Tanwin Fath
                                ُ    | # Damma
                                ٌ    | # Tanwin Damm
                                ِ    | # Kasra
                                ٍ    | # Tanwin Kasr
                                ْ    | # Sukun
                                ـ     # Tatwil/Kashida
                            """, re.VERBOSE)
        #normalize
        sentence = re.sub(arabic_diacritics, '', sentence)
        sentence = re.sub("[إأآا]", "ا", sentence)
        sentence = re.sub("ى", "ي", sentence)
        sentence = re.sub("ة", "ه", sentence)
        return sentence

    def clean(self, sentences):
        if self.lang == 'ar':
            return [self.ar_clean(sen) for sen in sentences]
        else: return [self.en_clean(sen) for sen in sentences]

    def creat_tensors(self, l_sen_tkn):
        max_len = max([len(sen) for sen in l_sen_tkn])
        batch = len(l_sen_tkn)
        tensor_data = torch.zeros(batch, max_len, dtype=torch.long)
        tensor_mask = []
        for i in range(batch):
            ids = [self.word2index.get(w, self.word2index["<UNK>"]) for w in l_sen_tkn[i]]
            tensor_data[i, 0:len(ids)] = torch.tensor(ids, dtype=torch.long)
            tensor_mask.append(len(ids))
        return tensor_data, torch.tensor(tensor_mask)
