import math
from collections import Counter, defaultdict
import random
import nltk
from nltk.corpus import brown
import time


def build_models():
    words = [word.lower() for word in brown.words() if word.isalpha()]

    # Unigram frequencies
    unigram_counts = Counter(words)

    # Vocabulary
    vocabulary = set(words)

    # Bigram counts
    bigram_counts = defaultdict(Counter)

    for w1, w2 in zip(words, words[1:]):
        bigram_counts[w1][w2] += 1

    return vocabulary, unigram_counts, bigram_counts


def bigram_probability(w1, w2, bigram_counts, unigram_counts, k=1.0):
    vocabulary_size = len(unigram_counts)

    count_bigram = bigram_counts[w1][w2]
    count_w1 = unigram_counts[w1]

    return (count_bigram + k) / (count_w1 + k * vocabulary_size)


def edits1(word):
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    deletions = [
        left + right[1:]
        for left, right in splits
        if right
    ]

    transposes = [
        left + right[1] + right[0] + right[2:]
        for left, right in splits
        if len(right) > 1
    ]

    replacements = [
        left + c + right[1:]
        for left, right in splits
        if right
        for c in alphabet
    ]

    insertions = [
        left + c + right
        for left, right in splits
        for c in alphabet
    ]

    return set(deletions + transposes + replacements + insertions)


def build_delete_dictionary(vocabulary):
    delete_dict = defaultdict(set)

    for word in vocabulary:
        for i in range(len(word)):
            deleted = word[:i] + word[i + 1:]
            delete_dict[deleted].add(word)

    return delete_dict


def symmetric_delete_candidates(word, delete_dict):
    candidates = set()

    for i in range(len(word)):
        deleted = word[:i] + word[i + 1:]
        candidates.update(delete_dict.get(deleted, set()))

    return candidates



# candidates = symmetric_delete_candidates("teh", delete_dict)
# print("Method B candidates:", candidates)
# print("the" in candidates)

def correct_nonword(word, vocabulary, unigram_counts, delete_dict):
    candidates_a = edits1(word)
    candidates_b = symmetric_delete_candidates(word, delete_dict)

    candidates = (candidates_a | candidates_b) & vocabulary

    if not candidates:
        return word

    return max(candidates, key=lambda w: unigram_counts[w])


def correct_realword(previous_word, word, next_word,
                     vocabulary, unigram_counts,
                     bigram_counts, delete_dict, k=1.0):
    candidates = (
        edits1(word)
        | symmetric_delete_candidates(word, delete_dict)
    ) & vocabulary

    best_word = word

    def score(candidate):
        score_value = bigram_probability(
            previous_word, candidate,
            bigram_counts, unigram_counts, k
        )

        if next_word in vocabulary:
            score_value *= bigram_probability(
                candidate, next_word,
                bigram_counts, unigram_counts, k
            )

        return score_value

    original_score = score(word)
    best_score = original_score

    for candidate in candidates:
        candidate_score = score(candidate)

        if candidate_score > best_score * 2:
            best_score = candidate_score
            best_word = candidate

    return best_word



# print("Non-word:", correct_nonword(
#     "teh", vocabulary, unigram_counts, delete_dict
# ))
# print("Real-word:", correct_realword(
#     "to",
#     "sea",
#     "the",
#     vocabulary,
#     unigram_counts,
#     bigram_counts,
#     delete_dict
# ))



def make_test_set(vocabulary, seed=42):
    random.seed(seed)

    sentences = [
        [word.lower() for word in sent if word.isalpha()]
        for sent in brown.sents()
    ]
    sentences = [sent for sent in sentences if len(sent) >= 3]

    num_sentences = max(1, int(0.10 * len(sentences)))

    selected = random.sample(
         sentences,
         num_sentences
    )

    test_data = []

    for sentence in selected:
        index = random.randrange(len(sentence))
        original = sentence[index]

        # Non-word error: create one edit and keep it only
        # if the corrupted form is not in the vocabulary.
        nonword_candidates = [
            candidate
            for candidate in edits1(original)
            if candidate not in vocabulary and candidate.isalpha()
        ]

        if not nonword_candidates:
            continue

        nonword = random.choice(nonword_candidates)

        # Real-word error: find a one-edit candidate that
        # is also a real vocabulary word.
        realword_candidates = [
            candidate
            for candidate in edits1(original)
            if candidate in vocabulary and candidate != original
        ]

        if not realword_candidates:
            continue

        realword = random.choice(realword_candidates)

        test_data.append({
            "sentence": sentence,
            "index": index,
            "original": original,
            "nonword": nonword,
            "realword": realword,
        })

    return test_data

def evaluate_test_set(test_data, vocabulary, unigram_counts,
                      bigram_counts, delete_dict):
    nonword_correct = 0
    realword_correct = 0

    for item in test_data:
        predicted_nonword = correct_nonword(
            item["nonword"],
            vocabulary,
            unigram_counts,
            delete_dict
        )

        if predicted_nonword == item["original"]:
            nonword_correct += 1

        previous_word = ""
        next_word = ""

        if item["index"] > 0:
            previous_word = item["sentence"][item["index"] - 1]

        if item["index"] + 1 < len(item["sentence"]):
            next_word = item["sentence"][item["index"] + 1]

        predicted_realword = correct_realword(
        previous_word,
        item["realword"],
        next_word,
        vocabulary,
        unigram_counts,
        bigram_counts,
        delete_dict
)

        if predicted_realword == item["original"]:
            realword_correct += 1

    total = len(test_data)

    return {
        "nonword_accuracy": nonword_correct / total if total else 0,
        "realword_accuracy": realword_correct / total if total else 0,
    }

def speed_benchmark(vocabulary, delete_dict, n=1000):
    test_words = [
        word for word in vocabulary
        if len(word) >= 3
    ]

    random.seed(42)
    selected = random.sample(test_words, n)

    misspelled_words = []
    for word in selected:
        edits = [
            candidate
            for candidate in edits1(word)
            if candidate.isalpha()
        ]
        if edits:
            misspelled_words.append(random.choice(edits))

    misspelled_words = misspelled_words[:n]

    start = time.perf_counter()
    for word in misspelled_words:
        edits1(word)
    method_a_time = time.perf_counter() - start

    start = time.perf_counter()
    for word in misspelled_words:
        symmetric_delete_candidates(word, delete_dict)
    method_b_time = time.perf_counter() - start

    print(f"Benchmark words: {len(misspelled_words)}")
    print(f"Method A time: {method_a_time:.6f} seconds")
    print(f"Method B time: {method_b_time:.6f} seconds")

    return method_a_time, method_b_time


def interactive_cli(vocabulary, unigram_counts, bigram_counts, delete_dict):
    print("Spelling Corrector")
    print("Type 'exit' to quit.")

    while True:
        sentence = input("\nEnter a sentence: ").strip()

        if sentence.lower() == "exit":
            print("Goodbye!")
            break

        start = time.perf_counter()

        words = sentence.split()
        corrected_words = []

        for i, original_token in enumerate(words):
            prefix = ""
            suffix = ""
            word = original_token

            while word and not word[0].isalnum():
                prefix += word[0]
                word = word[1:]

            while word and not word[-1].isalnum():
                suffix = word[-1] + suffix
                word = word[:-1]

            clean_word = word.lower()

            if not clean_word:
                corrected_words.append(original_token)
                continue

            if clean_word not in vocabulary:
                corrected = correct_nonword(
                    clean_word,
                    vocabulary,
                    unigram_counts,
                    delete_dict
                )
            else:
                previous_word = ""
                next_word = ""

                if i > 0:
                    previous_word = words[i - 1].strip(
                        ".,!?;:\"'()[]{}"
                    ).lower()

                if i + 1 < len(words):
                    next_word = words[i + 1].strip(
                        ".,!?;:\"'()[]{}"
                    ).lower()

                if previous_word in vocabulary:
                    corrected = correct_realword(
                        previous_word,
                        clean_word,
                        next_word,
                        vocabulary,
                        unigram_counts,
                        bigram_counts,
                        delete_dict
                    )
                else:
                    corrected = clean_word

            if corrected != clean_word:
                corrected_words.append(
                    f"{prefix}**{corrected}**{suffix}"
                )
            else:
                corrected_words.append(original_token)

        corrected_sentence = " ".join(corrected_words)

        latency = (time.perf_counter() - start) * 1000

        print("Corrected:", corrected_sentence)
        print(f"Latency: {latency:.2f} ms")

def main():
    vocabulary, unigram_counts, bigram_counts = build_models()
    delete_dict = build_delete_dictionary(vocabulary)

    test_data = make_test_set(vocabulary)

    results = evaluate_test_set(
        test_data,
        vocabulary,
        unigram_counts,
        bigram_counts,
        delete_dict
    )

    print("Non-word accuracy:", results["nonword_accuracy"])
    print("Real-word accuracy:", results["realword_accuracy"])

    interactive_cli(
        vocabulary,
        unigram_counts,
        bigram_counts,
        delete_dict
    )


if __name__ == "__main__":
    main()