import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # TODO implement the recognizer
    for x in  test_set.get_all_sequences():
        temp_dict = {}
        for word,model in models.items():
            try:
                X,lengths = test_set.get_item_Xlengths(x)
                temp_dict[word] = model.score(X,lengths)
            except:
                temp_dict[word] = float('-inf')
        if temp_dict:
            probabilities.append(dict(temp_dict))
            guesses.append(max(temp_dict, key=temp_dict.get))
    return probabilities,guesses
    raise NotImplementedError

