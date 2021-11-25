import numpy as np

WORD_FORMS = [
    "DET", "NOUN", "ADP", "VERB", "ADJ", "PUNCT",
    "SCONJ", "PRON", "AUX", "CCONJ", "PROPN", "ADV",
    "NUM", "PART", "INTJ", "SYM", "X", "_"
]


def number_of_sentences(sentences):
    """ Checked, works """
    return len(sentences)


def mean_length_of_a_sentence(sentences):
    """ Checked, works """
    return sum([len(sentence) for sentence in sentences]) / len(sentences)


def variance_length_of_a_sentence(sentences):
    """ Checked, works """
    return np.var(np.array([[len(sentence)] for sentence in sentences]))


def vocab_size(sentences):
    """ Checked, works """
    return len({word["lemma"] for sentence in sentences for word in sentence})


###################################################
# EN DESSOUS DE CETTE LIGNE GIT DU CODE NON TESTE #
###################################################


def mean_distance_to_head(sentences):
    word_count = 0
    distance_to_head = 0

    for sentence in sentences:
        word_count += len(sentence)

        for word in sentence:
            if isinstance(word['id'], int) and isinstance(word['head'], int):
                distance_to_head += (abs(word['id'] - word['head']))

    return distance_to_head / word_count


def max_nested_dependency(sentences):
    nb_sentences = 0
    max_total = 0

    for sentence in sentences:
        nb_sentences += 1
        max_len = 0
        for word in sentence:
            current_word = word
            current_len = 0
            while current_word["head"] is not None and current_word["head"] > 0:
                if current_word["head"] - 1 == current_word["id"]:
                    break
                current_word = sentence[current_word["head"] - 1]
                current_len += 1
                if current_len >= len(sentence):
                    current_len = 0
                    break
            if current_len > max_len:
                max_len = current_len
        max_total += max_len
    return max_total / nb_sentences


def var_distance_to_head(sentences):
    distances_to_head = []
    for sentence in sentences:
        for word in sentence:
            if isinstance(word['id'], int) and isinstance(word['head'], int):
                distances_to_head += [(abs(word['id'] - word['head']))]

    return np.var(np.array(distances_to_head))


def var_max_nested_dependency(sentences):
    nb_sentences = 0
    maxs_total = []

    for sentence in sentences:
        nb_sentences += 1
        max_len = 0
        for word in sentence:
            current_word = word
            current_len = 0
            while current_word["head"] is not None and current_word["head"] > 0:
                if current_word["head"] - 1 == current_word["id"]:
                    break
                current_word = sentence[current_word["head"] - 1]
                current_len += 1
                if current_len >= len(sentence):
                    current_len = 0
                    break
            if current_len > max_len:
                max_len = current_len
        maxs_total += [max_len]
    return np.var(np.array(maxs_total))


def root_std(sentences):
    """
    Standard deviation of the position of the root in the sentence.
    """
    positions = []
    for sentence in sentences:
        for word in sentence:
            if word["head"] == 0:
                positions += [word["id"] / len(sentence)]
    positions = np.array(positions)

    return np.sqrt(np.sum((positions - positions.mean()) ** 2))


def non_projective(sentences):
    """
    Proportion of non-projective sentences.
    """
    nb_sentences = 0
    non_proj = 0
    for sentence in sentences:
        nb_sentences += 1
        arcs = []
        turbo_break = False
        for word in sentence:
            if word["head"] and word["id"]:
                arcs += [(word["head"], word["id"])]
        for i in range(len(arcs) - 1):
            for j in range(i, len(arcs)):
                min_j = min(arcs[j][0], arcs[j][1])
                max_j = max(arcs[j][0], arcs[j][1])
                # arcs[i][0] inside arcs[j]
                if arcs[i][0] > min_j and arcs[i][0] < max_j:
                    # arcs[i][1] outside arcs[j]
                    if arcs[i][1] < min_j or arcs[i][1] > max_j:
                        non_proj += 1
                        turbo_break = True
                        break
            if turbo_break:
                break
    return non_proj / nb_sentences


def cycle_per_sentence(sentences):
    """
    FIXME
    Number of dependency cycle over number of sentences.
    """
    cycle_count = 0
    nb_sentences = 0

    print('cycle')

    turbo_break = False
    for sentence in sentences:
        nb_sentences += 1
        for word in sentence:
            current_word = word
            current_len = 0
            visited = []
            while current_word["head"] is not None and current_word["head"] > 0:
                visited += [current_word["id"]]
                if current_word["head"] in visited:
                    cycle_count += 1
                    turbo_break = True
                    break
                current_word = sentence[current_word["head"]]
            if turbo_break:
                turbo_break = False
                break

    print('cycle ok')
    return cycle_count / nb_sentences


def num_each_form(corpus):
    upos_dic = dict()
    nb_sentences = 0

    for sentence in corpus:
        nb_sentences += 1
        for word in sentence:
            if word["upostag"] in upos_dic:
                upos_dic[word["upostag"]] += 1
            else:
                upos_dic[word["upostag"]] = 1

    res = np.zeros(len(WORD_FORMS))
    word_forms = np.array(WORD_FORMS)
    for k, v in upos_dic.items():
        res[np.where(word_forms == k)[0]] = v / nb_sentences
    return res
