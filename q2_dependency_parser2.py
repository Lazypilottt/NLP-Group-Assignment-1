import os
import time
from dataclasses import dataclass
from collections import defaultdict, Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Token:
    id: int
    form: str
    upos: str
    head: int
    deprel: str


@dataclass
class Sentence:
    tokens: list


@dataclass
class Configuration:
    stack: list
    buffer: list
    arcs: list


# ============================================================
# PART 1: CONLL-U READER
# ============================================================

def read_conllu(file_path):
    """
    Read a CoNLL-U file.

    Each token is stored as:
        ID, FORM, UPOS, HEAD, DEPREL

    Multi-word token lines such as 1-2 and empty nodes such as 3.1
    are ignored.
    """

    sentences = []
    current_tokens = []

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            # Blank line = end of sentence
            if not line:
                if current_tokens:
                    sentences.append(Sentence(current_tokens))
                    current_tokens = []
                continue

            # Ignore comments
            if line.startswith("#"):
                continue

            columns = line.split("\t")

            if len(columns) < 8:
                continue

            token_id = columns[0]

            # Ignore multi-word tokens and empty nodes
            if "-" in token_id or "." in token_id:
                continue

            try:
                token_id = int(token_id)
                head = int(columns[6])
            except ValueError:
                continue

            form = columns[1]
            upos = columns[3]
            deprel = columns[7]

            token = Token(
                id=token_id,
                form=form,
                upos=upos,
                head=head,
                deprel=deprel
            )

            current_tokens.append(token)

    # Handle last sentence
    if current_tokens:
        sentences.append(Sentence(current_tokens))

    return sentences


# ============================================================
# TOKEN / CONFIGURATION HELPERS
# ============================================================

def make_token_dictionary(sentence):
    """
    Create dictionary:
        token ID -> Token

    Token 0 represents ROOT.
    """

    token_dict = {
        0: Token(
            id=0,
            form="<ROOT>",
            upos="ROOT",
            head=-1,
            deprel="root"
        )
    }

    for token in sentence.tokens:
        token_dict[token.id] = token

    return token_dict


def initialize_configuration(sentence):
    """
    Initial arc-standard configuration:

        Stack  = [ROOT]
        Buffer = [1, 2, ..., n]
        Arcs   = []
    """

    token_ids = [token.id for token in sentence.tokens]

    return Configuration(
        stack=[0],
        buffer=token_ids.copy(),
        arcs=[]
    )


# ============================================================
# ARC-STANDARD TRANSITIONS
# ============================================================

def apply_shift(config):
    """
    SHIFT:
        Move first item of buffer to top of stack.
    """

    if not config.buffer:
        return False

    token = config.buffer.pop(0)
    config.stack.append(token)

    return True


def apply_left_arc(config, label):
    """
    LEFT-ARC(label):

        Stack: ... second_top, top

        Add:
            top -> second_top

        Then remove second_top.
    """

    if len(config.stack) < 2:
        return False

    second_top = config.stack[-2]
    top = config.stack[-1]

    # ROOT cannot be a dependent
    if second_top == 0:
        return False

    config.arcs.append(
        (top, second_top, label)
    )

    config.stack.pop(-2)

    return True


def apply_right_arc(config, label):
    """
    RIGHT-ARC(label):

        Stack: ... second_top, top

        Add:
            second_top -> top

        Then remove top.
    """

    if len(config.stack) < 2:
        return False

    second_top = config.stack[-2]
    top = config.stack[-1]

    # ROOT cannot be a dependent
    if top == 0:
        return False

    config.arcs.append(
        (second_top, top, label)
    )

    config.stack.pop()

    return True


# ============================================================
# GOLD TREE INFORMATION
# ============================================================

def build_gold_children(sentence):
    """
    Build:

        head -> set(children)

    Example:

        2 -> {1, 3, 5}
    """

    children = defaultdict(set)

    for token in sentence.tokens:
        children[token.head].add(token.id)

    return children


def all_children_attached(
    token_id,
    gold_children,
    attached_dependents
):
    """
    A token can be attached only after all of its gold children
    have already been attached.

    This is required for the arc-standard oracle.
    """

    required_children = gold_children.get(
        token_id,
        set()
    )

    return required_children.issubset(
        attached_dependents
    )


# ============================================================
# ORACLE
# ============================================================

def get_oracle_transition(
    config,
    token_dict,
    gold_children,
    attached_dependents
):
    """
    Static oracle for arc-standard parsing.

    Priority:

        1. LEFT-ARC
        2. RIGHT-ARC
        3. SHIFT

    A LEFT/RIGHT arc is performed only when the dependent's
    complete subtree has already been processed.
    """

    if len(config.stack) >= 2:

        second_top = config.stack[-2]
        top = config.stack[-1]

        # ----------------------------------------------------
        # LEFT-ARC
        # ----------------------------------------------------

        if second_top != 0:

            second_token = token_dict[second_top]

            if (
                second_token.head == top
                and all_children_attached(
                    second_top,
                    gold_children,
                    attached_dependents
                )
            ):

                return (
                    f"LEFT:{second_token.deprel}"
                )

        # ----------------------------------------------------
        # RIGHT-ARC
        # ----------------------------------------------------

        if top != 0:

            top_token = token_dict[top]

            if (
                top_token.head == second_top
                and all_children_attached(
                    top,
                    gold_children,
                    attached_dependents
                )
            ):

                return (
                    f"RIGHT:{top_token.deprel}"
                )

    # --------------------------------------------------------
    # SHIFT
    # --------------------------------------------------------

    if config.buffer:
        return "SHIFT"

    # No valid transition
    return None


# ============================================================
# PART 2: FEATURE EXTRACTION
# ============================================================

def extract_features(config, token_dict):
    """
    REQUIRED FEATURE SET FROM THE ASSIGNMENT:

        1. POS of top of stack
        2. POS of second stack item
        3. POS of first buffer item
        4. POS of second buffer item

    No lexical features are used.
    """

    features = {}

    # --------------------------------------------------------
    # Top of stack
    # --------------------------------------------------------

    if len(config.stack) >= 1:

        features["stack_top_pos"] = (
            token_dict[
                config.stack[-1]
            ].upos
        )

    else:

        features["stack_top_pos"] = "<NULL>"

    # --------------------------------------------------------
    # Second item on stack
    # --------------------------------------------------------

    if len(config.stack) >= 2:

        features["stack_second_pos"] = (
            token_dict[
                config.stack[-2]
            ].upos
        )

    else:

        features["stack_second_pos"] = "<NULL>"

    # --------------------------------------------------------
    # First buffer item
    # --------------------------------------------------------

    if len(config.buffer) >= 1:

        features["buffer_first_pos"] = (
            token_dict[
                config.buffer[0]
            ].upos
        )

    else:

        features["buffer_first_pos"] = "<NULL>"

    # --------------------------------------------------------
    # Second buffer item
    # --------------------------------------------------------

    if len(config.buffer) >= 2:

        features["buffer_second_pos"] = (
            token_dict[
                config.buffer[1]
            ].upos
        )

    else:

        features["buffer_second_pos"] = "<NULL>"

    return features


# ============================================================
# GENERATE TRAINING INSTANCES
# ============================================================

def generate_training_instances(sentence):
    """
    Simulate the gold parse using the oracle.

    For every configuration:

        features -> correct transition

    is generated.
    """

    token_dict = make_token_dictionary(sentence)

    config = initialize_configuration(sentence)

    gold_children = build_gold_children(sentence)

    # Tokens that have already received their gold head
    attached_dependents = set()

    feature_list = []
    label_list = []

    maximum_steps = (
        4 * len(sentence.tokens) + 20
    )

    steps = 0

    while (
        config.buffer
        or len(config.stack) > 1
    ):

        steps += 1

        # Safety against infinite loops
        if steps > maximum_steps:

            return [], [], False, "maximum_steps"

        # Extract features
        features = extract_features(
            config,
            token_dict
        )

        # Get gold transition
        transition = get_oracle_transition(
            config,
            token_dict,
            gold_children,
            attached_dependents
        )

        # No oracle transition means the tree cannot be
        # completed using this transition system.
        if transition is None:

            return [], [], False, "non_projective_oracle_deadend"

        feature_list.append(features)
        label_list.append(transition)

        # ----------------------------------------------------
        # APPLY SHIFT
        # ----------------------------------------------------

        if transition == "SHIFT":

            success = apply_shift(config)

            if not success:
                return [], [], False, "invalid_shift"

        # ----------------------------------------------------
        # APPLY LEFT-ARC
        # ----------------------------------------------------

        elif transition.startswith("LEFT:"):

            label = transition.split(
                ":", 1
            )[1]

            dependent = config.stack[-2]

            success = apply_left_arc(
                config,
                label
            )

            if not success:
                return [], [], False, "invalid_left_arc"

            attached_dependents.add(
                dependent
            )

        # ----------------------------------------------------
        # APPLY RIGHT-ARC
        # ----------------------------------------------------

        elif transition.startswith("RIGHT:"):

            label = transition.split(
                ":", 1
            )[1]

            dependent = config.stack[-1]

            success = apply_right_arc(
                config,
                label
            )

            if not success:
                return [], [], False, "invalid_right_arc"

            attached_dependents.add(
                dependent
            )

    return (
        feature_list,
        label_list,
        True,
        None
    )


# ============================================================
# CREATE COMPLETE TRAINING DATASET
# ============================================================

def create_training_data(sentences):
    """
    Generate classifier training data from all training
    sentences.
    """

    all_features = []
    all_labels = []

    successful_sentences = 0
    skipped_sentences = 0

    skip_reasons = Counter()

    print(
        "\nGenerating oracle training instances..."
    )

    for index, sentence in enumerate(sentences):

        (
            features,
            labels,
            success,
            reason
        ) = generate_training_instances(sentence)

        if success:

            all_features.extend(features)
            all_labels.extend(labels)

            successful_sentences += 1

        else:

            skipped_sentences += 1

            if reason:
                skip_reasons[reason] += 1

        # Progress
        if (index + 1) % 1000 == 0:

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(sentences)} sentences"
            )

    print(
        "\nOracle generation completed."
    )

    print(
        "Successful sentences :",
        successful_sentences
    )

    print(
        "Skipped sentences    :",
        skipped_sentences
    )

    print(
        "Training instances   :",
        len(all_features)
    )

    if skip_reasons:

        print(
            "\nSkip reasons:"
        )

        for reason, count in skip_reasons.items():

            print(
                f"  {reason}: {count}"
            )

    return (
        all_features,
        all_labels
    )


# ============================================================
# TRANSITION STATISTICS
# ============================================================

def print_transition_statistics(labels):
    """
    Print the distribution of transition classes.
    """

    counts = Counter(labels)

    print(
        "\nTransition statistics:"
    )

    print(
        "SHIFT      :",
        sum(
            count
            for label, count in counts.items()
            if label == "SHIFT"
        )
    )

    print(
        "LEFT-ARC   :",
        sum(
            count
            for label, count in counts.items()
            if label.startswith("LEFT:")
        )
    )

    print(
        "RIGHT-ARC  :",
        sum(
            count
            for label, count in counts.items()
            if label.startswith("RIGHT:")
        )
    )

    print(
        "Unique labeled transitions:",
        len(counts)
    )


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

def train_classifier(features, labels):
    print("\nVectorizing features...")

    vectorizer = DictVectorizer(sparse=True)

    X = vectorizer.fit_transform(features)

    # Required for compatibility with some
    # SciPy / scikit-learn versions.
    X.indices = X.indices.astype("int32")
    X.indptr = X.indptr.astype("int32")

    print("Feature matrix shape:", X.shape)
    print("Sparse matrix index dtype:", X.indices.dtype)
    print("Sparse matrix indptr dtype:", X.indptr.dtype)

    print("\nTraining classifier...")

    model = SGDClassifier(
        loss="log_loss",
        max_iter=100,
        tol=1e-3,
        random_state=42
    )

    start_time = time.time()

    model.fit(X, labels)

    end_time = time.time()

    print(
        f"Training completed in "
        f"{end_time - start_time:.2f} seconds"
    )

    print(
        "Number of transition classes:",
        len(model.classes_)
    )

    return vectorizer, model


# ============================================================
# TRANSITION VALIDITY
# ============================================================

def is_valid_transition(
    transition,
    config
):
    """
    Check whether a predicted transition can legally be
    applied to the current configuration.
    """

    # --------------------------------------------------------
    # SHIFT
    # --------------------------------------------------------

    if transition == "SHIFT":

        return (
            len(config.buffer) > 0
        )

    # --------------------------------------------------------
    # LEFT-ARC
    # --------------------------------------------------------

    if transition.startswith("LEFT:"):

        if len(config.stack) < 2:
            return False

        # ROOT cannot be dependent
        if config.stack[-2] == 0:
            return False

        return True

    # --------------------------------------------------------
    # RIGHT-ARC
    # --------------------------------------------------------

    if transition.startswith("RIGHT:"):

        if len(config.stack) < 2:
            return False

        # ROOT cannot be dependent
        if config.stack[-1] == 0:
            return False

        return True

    return False


# ============================================================
# PREDICT NEXT TRANSITION
# ============================================================

def predict_transition(
    config,
    token_dict,
    vectorizer,
    model
):
    """
    Predict the highest-probability legal transition.

    The classifier may predict an illegal action, so we rank
    all predictions by probability and choose the first legal
    one.
    """

    features = extract_features(
        config,
        token_dict
    )

    X = vectorizer.transform(
        [features]
    )

    probabilities = model.predict_proba(
        X
    )[0]

    ranked_indices = probabilities.argsort()[
        ::-1
    ]

    for index in ranked_indices:

        transition = model.classes_[
            index
        ]

        if is_valid_transition(
            transition,
            config
        ):

            return transition

    # --------------------------------------------------------
    # Defensive fallback
    # --------------------------------------------------------

    if config.buffer:

        return "SHIFT"

    if len(config.stack) >= 2:

        # Find a known legal RIGHT transition
        for transition in model.classes_:

            if (
                transition.startswith("RIGHT:")
                and is_valid_transition(
                    transition,
                    config
                )
            ):

                return transition

        # Last-resort generic dependency label
        return "RIGHT:dep"

    return None


# ============================================================
# PARSE SENTENCE
# ============================================================

def parse_sentence(
    sentence,
    vectorizer,
    model
):
    """
    Parse one sentence using the trained classifier.
    """

    token_dict = make_token_dictionary(
        sentence
    )

    config = initialize_configuration(
        sentence
    )

    maximum_steps = (
        4 * len(sentence.tokens) + 20
    )

    steps = 0

    while (
        config.buffer
        or len(config.stack) > 1
    ):

        steps += 1

        if steps > maximum_steps:

            print(
                "WARNING: Maximum parser "
                "steps exceeded."
            )

            break

        transition = predict_transition(
            config,
            token_dict,
            vectorizer,
            model
        )

        if transition is None:
            break

        # ----------------------------------------------------
        # SHIFT
        # ----------------------------------------------------

        if transition == "SHIFT":

            apply_shift(config)

        # ----------------------------------------------------
        # LEFT-ARC
        # ----------------------------------------------------

        elif transition.startswith(
            "LEFT:"
        ):

            label = transition.split(
                ":",
                1
            )[1]

            apply_left_arc(
                config,
                label
            )

        # ----------------------------------------------------
        # RIGHT-ARC
        # ----------------------------------------------------

        elif transition.startswith(
            "RIGHT:"
        ):

            label = transition.split(
                ":",
                1
            )[1]

            apply_right_arc(
                config,
                label
            )

    return config.arcs


# ============================================================
# ARC CONVERSION
# ============================================================

def arcs_to_dictionary(arcs):
    """
    Convert:

        [(head, dependent, label), ...]

    into:

        dependent -> (head, label)
    """

    predicted = {}

    for head, dependent, label in arcs:

        predicted[dependent] = (
            head,
            label
        )

    return predicted


# ============================================================
# EVALUATION
# ============================================================

def evaluate_las(
    sentences,
    vectorizer,
    model
):
    """
    Evaluate on the development set.

    UAS:
        Correct head / total tokens

    LAS:
        Correct head AND dependency label / total tokens
    """

    correct = 0
    head_correct = 0
    total = 0

    print(
        "\nEvaluating parser on development set..."
    )

    start_time = time.time()

    for index, sentence in enumerate(
        sentences
    ):

        predicted_arcs = parse_sentence(
            sentence,
            vectorizer,
            model
        )

        predicted = arcs_to_dictionary(
            predicted_arcs
        )

        for token in sentence.tokens:

            total += 1

            if token.id not in predicted:
                continue

            (
                predicted_head,
                predicted_label
            ) = predicted[token.id]

            # Correct head
            if predicted_head == token.head:

                head_correct += 1

                # Correct head + dependency label
                if (
                    predicted_label
                    == token.deprel
                ):

                    correct += 1

        if (index + 1) % 500 == 0:

            print(
                f"Evaluated "
                f"{index + 1}/"
                f"{len(sentences)} sentences"
            )

    end_time = time.time()

    if total == 0:

        return 0.0

    uas = (
        head_correct / total
    )

    las = (
        correct / total
    )

    print(
        "\n=============================="
    )

    print(
        "EVALUATION RESULTS"
    )

    print(
        "=============================="
    )

    print(
        "Total tokens:",
        total
    )

    print(
        "Correct heads:",
        head_correct
    )

    print(
        "Correct head + label:",
        correct
    )

    print(
        f"UAS: {uas * 100:.2f}%"
    )

    print(
        f"LAS: {las * 100:.2f}%"
    )

    print(
        f"Evaluation time: "
        f"{end_time - start_time:.2f} seconds"
    )

    return las


# ============================================================
# EXAMPLE SENTENCES
# ============================================================

def get_example_sentences():
    """
    Required inference examples from the assignment.

    Gold heads and labels are intentionally not supplied because
    these sentences are being used only to demonstrate parser
    predictions.
    """

    examples = []

    # --------------------------------------------------------
    # Example 1
    # The cat sat on the mat.
    # --------------------------------------------------------

    examples.append(
        Sentence([
            Token(1, "The", "DET", 0, ""),
            Token(2, "cat", "NOUN", 0, ""),
            Token(3, "sat", "VERB", 0, ""),
            Token(4, "on", "ADP", 0, ""),
            Token(5, "the", "DET", 0, ""),
            Token(6, "mat", "NOUN", 0, ""),
            Token(7, ".", "PUNCT", 0, "")
        ])
    )

    # --------------------------------------------------------
    # Example 2
    # She eats a green salad.
    # --------------------------------------------------------

    examples.append(
        Sentence([
            Token(1, "She", "PRON", 0, ""),
            Token(2, "eats", "VERB", 0, ""),
            Token(3, "a", "DET", 0, ""),
            Token(4, "green", "ADJ", 0, ""),
            Token(5, "salad", "NOUN", 0, ""),
            Token(6, ".", "PUNCT", 0, "")
        ])
    )

    # --------------------------------------------------------
    # Example 3
    # I saw the man with a telescope.
    # --------------------------------------------------------

    examples.append(
        Sentence([
            Token(1, "I", "PRON", 0, ""),
            Token(2, "saw", "VERB", 0, ""),
            Token(3, "the", "DET", 0, ""),
            Token(4, "man", "NOUN", 0, ""),
            Token(5, "with", "ADP", 0, ""),
            Token(6, "a", "DET", 0, ""),
            Token(7, "telescope", "NOUN", 0, ""),
            Token(8, ".", "PUNCT", 0, "")
        ])
    )

    return examples


# ============================================================
# PRINT PARSE
# ============================================================

def print_parse(
    sentence,
    arcs
):
    """
    Print dependency arcs in readable form.
    """

    token_dict = make_token_dictionary(
        sentence
    )

    print(
        "\nSentence:"
    )

    print(
        " ".join(
            token.form
            for token in sentence.tokens
        )
    )

    print(
        "\nPredicted dependency arcs:"
    )

    if not arcs:

        print(
            "No arcs predicted."
        )

        return

    # Sort by dependent ID
    arcs = sorted(
        arcs,
        key=lambda x: x[1]
    )

    for head, dependent, label in arcs:

        head_form = token_dict[
            head
        ].form

        dependent_form = token_dict[
            dependent
        ].form

        print(
            f"{dependent_form:<15}"
            f"--{label:<12}"
            f"--> {head_form}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=========================================="
    )

    print(
        "TRANSITION-BASED DEPENDENCY PARSER"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # Dataset paths
    # --------------------------------------------------------

    TRAIN_FILE = os.path.join(
        "UD_English-EWT",
        "en_ewt-ud-train.conllu"
    )

    DEV_FILE = os.path.join(
        "UD_English-EWT",
        "en_ewt-ud-dev.conllu"
    )

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not os.path.exists(TRAIN_FILE):

        print(
            "\nERROR: Training file not found:"
        )

        print(
            TRAIN_FILE
        )

        print(
            "\nMake sure the UD_English-EWT "
            "folder is in the same directory "
            "as this Python file."
        )

        return

    if not os.path.exists(DEV_FILE):

        print(
            "\nERROR: Development file not found:"
        )

        print(
            DEV_FILE
        )

        print(
            "\nMake sure the UD_English-EWT "
            "folder is in the same directory "
            "as this Python file."
        )

        return

    # --------------------------------------------------------
    # Read training data
    # --------------------------------------------------------

    print(
        "\nLoading training data..."
    )

    start_time = time.time()

    train_sentences = read_conllu(
        TRAIN_FILE
    )

    print(
        "Training sentences:",
        len(train_sentences)
    )

    # --------------------------------------------------------
    # Read development data
    # --------------------------------------------------------

    print(
        "\nLoading development data..."
    )

    dev_sentences = read_conllu(
        DEV_FILE
    )

    print(
        "Development sentences:",
        len(dev_sentences)
    )

    print(
        f"Loading time: "
        f"{time.time() - start_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Generate oracle training data
    # --------------------------------------------------------

    (
        features,
        labels
    ) = create_training_data(
        train_sentences
    )

    if not features or not labels:

        print(
            "\nERROR: No training data was generated."
        )

        return

    # --------------------------------------------------------
    # Transition statistics
    # --------------------------------------------------------

    print_transition_statistics(
        labels
    )

    # --------------------------------------------------------
    # Train classifier
    # --------------------------------------------------------

    vectorizer, model = train_classifier(
        features,
        labels
    )

    # --------------------------------------------------------
    # Evaluate on development set
    # --------------------------------------------------------

    las = evaluate_las(
        dev_sentences,
        vectorizer,
        model
    )

    # --------------------------------------------------------
    # Required example sentences
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "REQUIRED EXAMPLE SENTENCES"
    )

    print(
        "=========================================="
    )

    examples = get_example_sentences()

    for index, sentence in enumerate(
        examples,
        start=1
    ):

        print(
            f"\nExample {index}"
        )

        arcs = parse_sentence(
            sentence,
            vectorizer,
            model
        )

        print_parse(
            sentence,
            arcs
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=========================================="
    )

    print(
        f"Development LAS: "
        f"{las * 100:.2f}%"
    )

    print(
        "\nParser execution completed."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()