import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.translate.bleu_score import corpus_bleu

from gensim.models import KeyedVectors


WMD_model = KeyedVectors.load_word2vec_format('/home/xyf/LanguageModel/GoogleNews-vectors-negative300.bin.gz', binary=True)

stop_words = set(stopwords.words("english"))
tokenized_context_and_questions, untokenized_context_and_questions = [], []

sent1 = "ever since joes has changed hands it 's just gotten ."
sent2 = "ever since joes has changed hands it 's just gotten better and better ."

seg_sent1 = word_tokenize(sent1)
seg_sent2 = word_tokenize(sent2)
similarity_distance = WMD_model.wmdistance(seg_sent1, seg_sent2)


print('similarity_distance:', similarity_distance)
