# own modules
from features_list import *

FEATURES = [
    ("number_of_sentences", number_of_sentences),
    ("mean_length_of_a_sentence", mean_length_of_a_sentence),
    ("variance_length_of_a_sentence", variance_length_of_a_sentence),
    ("vocab_size", vocab_size),
    ("mean_distance_to_head", mean_distance_to_head),
    ("max_nested_dependency", max_nested_dependency),
    ("var_distance_to_head", var_distance_to_head),
    ("var_max_nested_dependency", var_max_nested_dependency),
    ("root_std", root_std),
    ("non_projective", non_projective)
    # FIXME ("cycle_per_sentence", cycle_per_sentence)
]

FEATURE_NAMES = [x[0] for x in FEATURES] + WORD_FORMS


def extract_features(sentences):
    n_word_forms = len(WORD_FORMS)
    n_features = len(FEATURES) + n_word_forms

    # initialize our features to 0
    features = np.zeros(n_features)

    # get the feature functions
    feature_functions = [feature[1] for feature in FEATURES]

    # apply each feature function !
    features[:n_features-n_word_forms] = [
        feature_function(sentences=sentences) for feature_function in feature_functions
    ]

    # apply the algorithm to count occurrences of each word form
    features[n_features-n_word_forms:n_features] = num_each_form(sentences)
    return features


if __name__ == "__main__":
    import pandas as pd
    from corpus import parse_from_conllu

    base_path = "/home/hippo/GoogleDrive/univ/master-2/TAL/projet/data"

    corpus_path = base_path + "/corpus"
    scores_path = base_path + "/scores.txt"

    scores = pd.read_csv(scores_path, sep=",", header=None)
    scores.columns = ['lang', 'las', 'uas']

    lang_codes = scores['lang']

    # ---
    # test with all languages
    # ---
    sentences, features = parse_from_conllu(pathname_dir=corpus_path, lang_codes=['en', 'fr'])

    df = pd.DataFrame(features, columns=FEATURE_NAMES)
    print(df)
