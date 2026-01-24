import random
import string
from copy import deepcopy
from statistics import mean

import pytest

import cipal

# learn --------------------------------------------------------------------------------


# Test that an empty LTM acquires all of the elements
def test_learn_elements():
    input = [" ".join(list(string.ascii_lowercase)[0:8])]
    ltm = cipal.new_ltm()
    cipal.learn(input, ltm)
    temp = cipal.ltm_to_df(ltm)
    assert list(temp["chunks"]) == ["a", "b", "c", "d", "e", "f", "g", "h"]


# Test that repeated presentation increases the speed of the learned chunks
def test_learn_elements_pt():
    input = [" ".join(list(string.ascii_lowercase)[0:8])]
    ltm = cipal.new_ltm()
    cipal.learn(input, ltm)
    temp1 = cipal.ltm_to_df(ltm)
    cipal.learn(input, ltm)
    temp2 = cipal.ltm_to_df(ltm)
    assert (
        list(temp1["chunks"])
        == list(temp2["chunks"])
        == ["a", "b", "c", "d", "e", "f", "g", "h"]
    )
    assert all([round(x) == 1175 for x in temp1["pt"]])
    assert all([round(x) == 1150 for x in temp2["pt"]])


# Test that CIPAL learns words when they are presented in different orders
def test_learn_words():
    w1 = "a b c"
    w2 = "d e f"
    w3 = "g h i"
    utts = [" ".join([w1, w2, w3]), " ".join([w3, w2, w1]), " ".join([w2, w3, w1])]
    ltm = cipal.new_ltm()
    ltm_len = 0
    for i in range(20):
        cipal.learn(utts, ltm)
        assert len(ltm) >= ltm_len  # LTM gets progressively larger
        ltm_len = len(ltm)
    assert w1 in ltm
    assert w2 in ltm
    assert w3 in ltm
    temp = cipal.ltm_to_df(ltm)
    assert list(temp["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "g h",
        "i d",
        "e f",
        "a b",
        "g h i",
        "a b c",
        "a b c d",
        "g h i d",
        "g h i d e f",
        "d e f",
        "a b c d e f",
    ]


# Test that the speech rate parameter changes the learning outcome
def test_learn_speech_rate():
    w1 = "a b c"
    w2 = "d e f"
    w3 = "g h i"
    utts = [" ".join([w1, w2, w3]), " ".join([w3, w2, w1]), " ".join([w2, w3, w1])]
    ltm1 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm1, speech_rate=150)
    ltm2 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm2, speech_rate=350)
    ltm3 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm3, speech_rate=550)
    temp1 = cipal.ltm_to_df(ltm1)
    temp2 = cipal.ltm_to_df(ltm2)
    temp3 = cipal.ltm_to_df(ltm3)
    assert list(temp1["chunks"]) != list(temp2["chunks"]) != list(temp3["chunks"])
    assert list(temp1["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "g h",
        "i d",
        "e f",
        "a b",
        "g h i",
        "a b c",
        "a b c d",
        "g h i d",
        "d e f",
        "a b c d e f",
        "g h i d e f",
        "a b c d e f g h i",
        "g h i d e f a b c",
        "d e f g h i",
        "d e f g h i a b c",
    ]
    assert list(temp2["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "a b",
        "c d",
        "e f",
        "g h",
        "g h i",
        "a b c",
        "a b c d",
        "g h i d",
        "a b c d e f",
        "g h i d e f",
        "d e f",
        "d e f g h",
        "g h i a b",
        "a b c d e f g h",
        "g h i d e f a b",
        "d e",
        "d e f g",
        "i a",
        "i a b",
        "a b c d e",
        "a b c d e f g",
        "a b c d e f g h i",
        "g h i d e",
        "g h i d e f a",
        "g h i d e f a b c",
        "i a b c",
    ]
    assert list(temp3["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "a b",
        "a b c",
        "a b c d",
        "a b c d e",
        "a b c d e f",
        "a b c d e f g",
        "a b c d e f g h",
        "a b c d e f g h i",
        "g h",
        "g h i",
        "g h i d",
        "g h i d e",
        "g h i d e f",
        "g h i d e f a",
        "d e",
        "d e f",
        "d e f g",
        "g h i a",
        "g h i d e f a b",
        "g h i d e f a b c",
        "d e f g h",
        "d e f g h i",
        "d e f g h i a",
        "d e f g h i a b",
        "d e f g h i a b c",
    ]
    assert round(mean(list(temp1["pt"]))) == 327
    assert round(mean(list(temp2["pt"]))) == 349
    assert round(mean(list(temp3["pt"]))) == 414


# Test that the decay rate parameter changes the learning outcome
def test_learn_decay():
    w1 = "a b c"
    w2 = "d e f"
    w3 = "g h i"
    utts = [" ".join([w1, w2, w3]), " ".join([w3, w2, w1]), " ".join([w2, w3, w1])]
    ltm1 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm1, decay_rate=640)
    ltm2 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm2, decay_rate=800)
    ltm3 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm3, decay_rate=1600)
    temp1 = cipal.ltm_to_df(ltm1)
    temp2 = cipal.ltm_to_df(ltm2)
    temp3 = cipal.ltm_to_df(ltm3)
    assert list(temp1["chunks"]) != list(temp2["chunks"]) != list(temp3["chunks"])
    assert list(temp1["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "d e",
        "f g",
        "h i",
        "a b",
        "a b c",
        "d e f",
        "d e f g",
        "d e f g h i",
        "g h i",
        "d e f g h i a b c",
        "d e f a b c",
        "a b c d e f",
        "g h i d e f",
        "g h i d e f a b c",
        "a b c d e f g",
        "a b c d e f g h i",
        "a b c d e",
        "g h i d e",
        "d e f g h i a b",
        "g h i d e f a b",
        "a b c d",
        "a b c d e f g h",
        "g h",
        "g h i d",
        "g h i d e f a",
        "d e f g h",
        "d e f g h i a",
    ]
    assert list(temp2["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "g h",
        "i d",
        "e f",
        "a b",
        "g h i",
        "a b c",
        "a b c d",
        "g h i d",
        "g h i d e f",
        "d e f",
        "a b c d e f",
        "a b c d e f g h i",
        "g h i d e f a b c",
        "d e f g h i",
        "d e f g h i a b c",
    ]
    assert list(temp3["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "a b",
        "c d",
        "e f",
        "g h",
        "g h i",
        "d e f",
        "a b c",
        "d e f g h i",
        "g h i d e f",
        "d e f g h i a b c",
        "g h i d e f a b c",
        "a b c d e f g h i",
        "a b c d e f",
        "a b c d",
        "g h i d",
    ]
    assert round(mean(list(temp1["pt"]))) == 165
    assert round(mean(list(temp2["pt"]))) == 279
    assert round(mean(list(temp3["pt"]))) == 472


# Test that increasing pt_adjust speeds up learning
def test_learn_pt_adjust():
    w1 = "a b c"
    w2 = "d e f"
    w3 = "g h i"
    utts = [" ".join([w1, w2, w3]), " ".join([w3, w2, w1]), " ".join([w2, w3, w1])]
    ltm1 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm1, pt_adjust=1)
    ltm2 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm2, pt_adjust=5)
    ltm3 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm3, pt_adjust=20)
    temp1 = cipal.ltm_to_df(ltm1)
    temp2 = cipal.ltm_to_df(ltm2)
    temp3 = cipal.ltm_to_df(ltm3)
    assert list(temp1["chunks"]) != list(temp2["chunks"]) != list(temp3["chunks"])
    assert list(temp1["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "d e",
        "f g",
        "h i",
        "a b",
        "a b c",
        "d e f",
        "d e f g",
        "d e f g h i",
        "g h i",
        "d e f g h i a b c",
        "d e f a b c",
    ]
    assert list(temp2["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "g h",
        "i d",
        "e f",
        "a b",
        "g h i",
        "a b c",
        "a b c d",
        "g h i d",
        "g h i d e f",
        "d e f",
        "a b c d e f",
        "a b c d e f g h i",
        "g h i d e f a b c",
        "d e f g h i",
        "d e f g h i a b c",
    ]
    assert list(temp3["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "g h",
        "i d",
        "e f",
        "a b",
        "g h i",
        "a b c",
        "a b c d",
        "g h i d",
        "d e f",
        "a b c d e f",
        "g h i d e f",
        "a b c d e f g h i",
        "g h i d e f a b c",
        "d e f g h i",
        "d e f g h i a b c",
        "g h i d e f a b",
        "d e f g h",
        "d e f g h i a b",
        "a b c d e f g h",
        "a b c d e",
        "a b c d e f g",
        "g h i d e",
        "g h i d e f a",
        "d e",
        "d e f g",
        "d e f g h i a",
    ]
    assert (
        len(list(temp1["chunks"]))
        < len(list(temp2["chunks"]))
        < len(list(temp3["chunks"]))
    )
    assert round(mean(list(temp1["pt"]))) == 511
    assert round(mean(list(temp2["pt"]))) == 279
    assert round(mean(list(temp3["pt"]))) == 112


# Test that the same results occur with/without the sign for pt_adjust
def test_learn_pt_adjust_sign():
    w1 = "a b c"
    w2 = "d e f"
    w3 = "g h i"
    utts = [" ".join([w1, w2, w3]), " ".join([w3, w2, w1]), " ".join([w2, w3, w1])]
    ltm1 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm1, pt_adjust=-1)
    ltm2 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm2, pt_adjust=-20)
    ltm3 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm3, pt_adjust=1)
    ltm4 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm4, pt_adjust=20)
    temp1 = cipal.ltm_to_df(ltm1)
    temp2 = cipal.ltm_to_df(ltm2)
    temp3 = cipal.ltm_to_df(ltm3)
    temp4 = cipal.ltm_to_df(ltm4)
    temp1.equals(temp3)
    temp2.equals(temp4)


# Test that a higher pt_initial slows down learning
def test_learn_pt_initial():
    w1 = "a b c"
    w2 = "d e f"
    w3 = "g h i"
    utts = [" ".join([w1, w2, w3]), " ".join([w3, w2, w1]), " ".join([w2, w3, w1])]
    ltm1 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm1, pt_initial=1000)
    ltm2 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm2, pt_initial=2000)
    ltm3 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm3, pt_initial=3000)
    temp1 = cipal.ltm_to_df(ltm1)
    temp2 = cipal.ltm_to_df(ltm2)
    temp3 = cipal.ltm_to_df(ltm3)
    assert list(temp1["chunks"]) != list(temp2["chunks"]) != list(temp3["chunks"])
    assert list(temp1["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "g h",
        "i d",
        "e f",
        "a b",
        "g h i",
        "a b c",
        "a b c d",
        "g h i d",
        "g h i d e f",
        "d e f",
        "a b c d e f",
        "d e f g h i",
        "d e f g h i a b c",
        "a b c d e f g h i",
        "g h i d e f a b c",
        "d e f g h",
        "d e f g h i a b",
        "a b c d e f g h",
        "g h i d e f a b",
        "a b c d e",
        "a b c d e f g",
        "g h i d e",
        "g h i d e f a",
        "d e",
        "d e f g",
        "d e f g h i a",
    ]
    assert list(temp2["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "d e",
        "f g",
        "h i",
        "a b",
        "a b c",
        "d e f",
        "d e f g",
        "g h i",
        "d e f g h i",
        "d e f a b c",
        "d e f g h i a b c",
        "g h i d e f",
        "g h i d e f a b c",
    ]
    assert list(temp3["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "d e",
        "f g",
        "h i",
        "a b",
        "a b c",
        "d e f",
        "d e f g",
        "d e f g h i",
        "g h i",
        "d e f a b c",
        "d e f g h i a b c",
    ]
    assert (
        len(list(temp1["chunks"]))
        > len(list(temp2["chunks"]))
        > len(list(temp3["chunks"]))
    )
    assert round(mean(list(temp1["pt"]))) == 218
    assert round(mean(list(temp2["pt"]))) == 345
    assert round(mean(list(temp3["pt"]))) == 448


# Test that the average pt is slower with a higher pt_ceiling
def test_learn_pt_ceiling():
    w1 = "a b c"
    w2 = "d e f"
    w3 = "g h i"
    utts = [" ".join([w1, w2, w3]), " ".join([w3, w2, w1]), " ".join([w2, w3, w1])]
    ltm1 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm1, pt_ceiling=100)
    ltm2 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm2, pt_ceiling=300)
    ltm3 = cipal.new_ltm()
    for i in range(100):
        cipal.learn(utts, ltm3, pt_ceiling=500)
    temp1 = cipal.ltm_to_df(ltm1)
    temp2 = cipal.ltm_to_df(ltm2)
    temp3 = cipal.ltm_to_df(ltm3)
    assert list(temp1["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "g h",
        "i d",
        "e f",
        "a b",
        "g h i",
        "a b c",
        "a b c d",
        "g h i d",
        "g h i d e f",
        "d e f",
        "a b c d e f",
        "a b c d e f g h i",
        "g h i d e f a b c",
        "d e f g h i",
        "d e f g h i a b c",
    ]
    assert list(temp2["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "g h",
        "i d",
        "e f",
        "a b",
        "g h i",
        "a b c",
        "a b c d",
        "g h i d",
        "g h i d e f",
        "d e f",
        "a b c d e f",
        "a b c d e f g h i",
        "g h i d e f a b c",
        "d e f g h i",
        "d e f g h i a b c",
    ]
    assert list(temp3["chunks"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "g h",
        "i d",
        "e f",
        "a b",
        "g h i",
        "a b c",
        "a b c d",
        "g h i d",
    ]
    assert round(mean(list(temp1["pt"]))) == 291
    assert round(mean(list(temp2["pt"]))) == 387
    assert round(mean(list(temp3["pt"]))) == 542


# Test that used chunks get faster but unused chunks remain the same
def test_learn_used_unused():
    w1 = "a b c"
    w2 = "d e f"
    ltm1 = cipal.new_ltm()
    for i in range(50):
        cipal.learn([w1, w2], ltm1)
    assert w1 in ltm1
    assert w2 in ltm1
    ltm2 = deepcopy(ltm1)
    for i in range(50):
        cipal.learn([w1], ltm2)
    assert ltm2["a b c"] < ltm1["a b c"]
    assert round(ltm1["a b c"]) == 365
    assert round(ltm2["a b c"]) == 69
    assert ltm2["d e f"] == ltm1["d e f"]
    assert round(ltm1["d e f"]) == round(ltm2["d e f"]) == 365


# process ------------------------------------------------------------------------------


# Test that processing items does not change LTM
def test_process_non_mutating():
    elements = list(string.ascii_lowercase)[0:24]
    w1 = " ".join(list(string.ascii_lowercase)[0:8])
    w2 = " ".join(list(string.ascii_lowercase)[8:16])
    w3 = " ".join(list(string.ascii_lowercase)[16:24])
    items = [w1, w2, w3]
    ltm = cipal.new_ltm()
    ltm.update({x: 100 for x in elements})
    temp1 = cipal.ltm_to_df(ltm)
    cipal.process(items, ltm)
    temp2 = cipal.ltm_to_df(ltm)
    assert temp1.equals(temp2)


# Test that CIPAL returns the elements
def test_process_elements():
    elements = list(string.ascii_lowercase)[0:24]
    w1 = " ".join(list(string.ascii_lowercase)[0:8])
    w2 = " ".join(list(string.ascii_lowercase)[8:16])
    w3 = " ".join(list(string.ascii_lowercase)[16:24])
    items = [w1, w2, w3]
    ltm = cipal.new_ltm()
    ltm.update({x: 100 for x in elements})
    temp = cipal.process(items, ltm)
    expected_parse = [
        "[a] [b] [c] [d] [e] [f] [g] [h]",
        "[i] [j] [k] [l] [m] [n] [o] [p]",
        "[q] [r] [s] [t] [u] [v] [w] [x]",
    ]
    assert list(temp["parse"]) == expected_parse
    assert all(x == 8 for x in temp["chunks"])
    assert all(x == 800 for x in temp["pt"])


# Test that CIPAL returns bigrams
def test_process_bigram():
    elements = list(string.ascii_lowercase)[0:24]
    w1 = " ".join(list(string.ascii_lowercase)[0:8])
    w2 = " ".join(list(string.ascii_lowercase)[8:16])
    w3 = " ".join(list(string.ascii_lowercase)[16:24])
    items = [w1, w2, w3]
    ltm = cipal.new_ltm()
    ltm.update({x: 100 for x in elements})
    bigrams = [
        "a b",
        "c d",
        "e f",
        "g h",
        "i j",
        "k l",
        "m n",
        "o p",
        "q r",
        "s t",
        "u v",
        "w x",
    ]
    ltm.update({x: 100 for x in bigrams})
    temp = cipal.process(items, ltm)
    expected_parse = [
        "[a b] [c d] [e f] [g h]",
        "[i j] [k l] [m n] [o p]",
        "[q r] [s t] [u v] [w x]",
    ]
    assert list(temp["parse"]) == expected_parse
    assert all(x == 4 for x in temp["chunks"])
    assert all(x == 400 for x in temp["pt"])


# Test that CIPAL returns the items
def test_process_items():
    elements = list(string.ascii_lowercase)[0:24]
    w1 = " ".join(list(string.ascii_lowercase)[0:8])
    w2 = " ".join(list(string.ascii_lowercase)[8:16])
    w3 = " ".join(list(string.ascii_lowercase)[16:24])
    items = [w1, w2, w3]
    ltm = cipal.new_ltm()
    ltm.update({x: 100 for x in elements})
    bigrams = [
        "a b",
        "c d",
        "e f",
        "g h",
        "i j",
        "k l",
        "m n",
        "o p",
        "q r",
        "s t",
        "u v",
        "w x",
    ]
    ltm.update({x: 100 for x in bigrams})
    ltm.update({x: 100 for x in items})
    temp = cipal.process(items, ltm)
    expected_parse = ["[a b c d e f g h]", "[i j k l m n o p]", "[q r s t u v w x]"]
    assert list(temp["parse"]) == expected_parse
    assert all(x == 1 for x in temp["chunks"])
    assert all(x == 100 for x in temp["pt"])


# Test that chunk selection is not affected by processing times
def test_process_pt():
    elements = list(string.ascii_lowercase)[0:24]
    w1 = " ".join(list(string.ascii_lowercase)[0:8])
    w2 = " ".join(list(string.ascii_lowercase)[8:16])
    w3 = " ".join(list(string.ascii_lowercase)[16:24])
    items = [w1, w2, w3]
    ltm1 = cipal.new_ltm()
    ltm2 = cipal.new_ltm()
    ltm1.update({x: 100 for x in elements})
    ltm1.update({x: 200 for x in items})
    ltm2.update({x: 200 for x in elements})
    ltm2.update({x: 100 for x in items})
    temp1 = cipal.process(items, ltm1)
    temp2 = cipal.process(items, ltm2)
    assert list(temp1["parse"]) == list(temp2["parse"])
    assert all(x == 200 for x in temp1["pt"])
    assert all(x == 100 for x in temp2["pt"])


# Test that CIPAL can parse a large list of (2000) items
def test_process_large_list():
    letters = list(string.ascii_lowercase)
    items = []
    for i in range(2000):
        items.append(" ".join(random.sample(letters, 10)))
    ltm = cipal.new_ltm()
    ltm.update({x: 100 for x in letters})
    temp = cipal.process(items, ltm)
    expected_parse = [" ".join([f"[{e}]" for e in itm.split()]) for itm in items]
    assert list(temp["parse"]) == expected_parse
    assert all(x == 10 for x in temp["chunks"])
    assert all(x == 1000 for x in temp["pt"])
    ltm.update({x: 100 for x in items})
    temp = cipal.process(items, ltm)
    expected_parse = [f"[{itm}]" for itm in items]
    assert list(temp["parse"]) == expected_parse
    assert all(x == 1 for x in temp["chunks"])
    assert all(x == 100 for x in temp["pt"])


# Test that an empty LTM throws an error
def test_process_empty_ltm():
    w1 = " ".join(list(string.ascii_lowercase)[0:8])
    w2 = " ".join(list(string.ascii_lowercase)[8:16])
    w3 = " ".join(list(string.ascii_lowercase)[16:24])
    items = [w1, w2, w3]
    ltm = cipal.new_ltm()
    with pytest.raises(ValueError):
        cipal.process(items, ltm)


# Test that items containing any unfamiliar elements throw an error
def test_process_unfamiliar():
    letters = list(string.ascii_lowercase)
    w1 = " ".join(list(string.ascii_lowercase)[0:8])
    w2 = " ".join(list(string.ascii_lowercase)[8:16])
    w3 = " ".join(list(string.ascii_lowercase)[16:24])
    items = [w1, w2, w3]
    ltm = cipal.new_ltm()
    ltm.update({x: 100 for x in letters[1:26]})
    with pytest.raises(ValueError):
        cipal.process(items, ltm)


# new_ltm ------------------------------------------------------------------------------


# Test that a new LTM hash table with no chunks is created
def test_new_ltm():
    ltm = cipal.new_ltm()
    assert ltm == {}
    assert len(ltm) == 0
    assert type(ltm) is dict
    ltm.update({x: 100 for x in list(string.ascii_lowercase)})
    assert len(ltm) == len(list(string.ascii_lowercase))


# new_stm ------------------------------------------------------------------------------


# Test that a new STM hash table is created
def test_new_stm():
    stm = cipal.new_stm()
    assert stm == {"chunks": [], "process": [], "decay": []}
    assert len(stm) == 3
    assert type(stm) is dict
    assert stm["chunks"] == []
    assert stm["process"] == []
    assert stm["decay"] == []


# check_stm ----------------------------------------------------------------------------


def test_check_stm():
    stm = cipal.new_stm()
    stm = {
        "chunks": ["a", "b", "c"],
        "process": [10, 20, 30],
        "decay": [100, 200, 300],
    }
    assert cipal.check_stm(stm) is None
    stm = {
        "chunks": ["a", "b"],
        "process": [10, 20, 30],
        "decay": [100, 200, 300],
    }
    with pytest.raises(ValueError):
        cipal.check_stm(stm)
    stm = {
        "chunks": ["a", "b", "c"],
        "process": [10, 20],
        "decay": [100, 200, 300],
    }
    with pytest.raises(ValueError):
        cipal.check_stm(stm)
    stm = {
        "chunks": ["a", "b", "c"],
        "process": [10, 20, 30],
        "decay": [100, 200],
    }
    with pytest.raises(ValueError):
        cipal.check_stm(stm)


# learn_element ------------------------------------------------------------------------


# Test that CIPAL adds unfamiliar elements to LTM with the initial pt
def test_learn_element_unfamiliar():
    ltm = cipal.new_ltm()
    cipal.learn_element("a", ltm, 1200)
    assert "a" in ltm
    assert ltm["a"] == 1200
    assert len(ltm) == 1
    cipal.learn_element("b", ltm, 10)
    assert "b" in ltm
    assert ltm["b"] == 10
    assert len(ltm) == 2
    cipal.learn_element("c", ltm, 500)
    assert "c" in ltm
    assert ltm["c"] == 500
    assert len(ltm) == 3


# Test that LTM does not change when familiar elements are presented
def test_learn_element_familiar():
    ltm = cipal.new_ltm()
    ltm["a"] = 100
    temp1 = cipal.ltm_to_df(ltm)
    cipal.learn_element("a", ltm, 1200)
    temp2 = cipal.ltm_to_df(ltm)
    assert temp1.equals(temp2)
    assert ltm["a"] == 100
    assert len(ltm) == 1


# add_to_stm ---------------------------------------------------------------------------


# Test that new elements are added to STM
def test_add_to_stm():
    tm_t = 160
    decay_rate = 800
    pt_initial = 1200
    ltm = cipal.new_ltm()
    ltm["a"] = pt_initial
    stm = cipal.new_stm()
    cipal.add_to_stm("a", stm, ltm, tm_t, decay_rate)
    assert stm["chunks"] == ["a"]
    assert stm["process"] == [tm_t + pt_initial]
    assert stm["decay"] == [tm_t + decay_rate]


# learn_chunks -------------------------------------------------------------------------


# Test that CIPAL does not learn from unprocessed sequences
def test_learn_chunks_unprocessed():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e"]
    ltm.update({x: 200 for x in elements})
    stm = {
        "chunks": ["a", "b", "c", "d", "e"],
        "process": list(range(300, 800, 100)),
        "decay": list(range(400, 900, 100)),
    }
    temp1 = cipal.ltm_to_df(ltm)
    cipal.learn_chunks(ltm, stm, 100)
    temp2 = cipal.ltm_to_df(ltm)
    assert temp1.equals(temp2)


# Test that CIPAL learns BC and DE chunks
def test_learn_chunks_bc_de():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e"]
    ltm.update({x: 200 for x in elements})
    stm = {
        "chunks": ["a", "b", "c", "d", "e"],
        "process": list(range(100, 600, 100)),
        "decay": list(range(200, 700, 100)),
    }
    cipal.learn_chunks(ltm, stm, 1000)
    assert "b c" in ltm
    assert "d e" in ltm
    assert len(ltm) == 7


# Test that a second presentation leads to AB and CD chunks
def test_learn_chunks_ab_cd():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e"]
    ltm.update({x: 200 for x in elements})
    stm = {
        "chunks": ["a", "b", "c", "d", "e"],
        "process": list(range(100, 600, 100)),
        "decay": list(range(200, 700, 100)),
    }
    for i in range(2):
        cipal.learn_chunks(ltm, stm, 1000)
    assert "a b" in ltm
    assert "c d" in ltm
    assert len(ltm) == 9


# Test that BC and DE are still used to recode the input
def test_learn_chunks_bc_de_recode():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e"]
    ltm.update({x: 200 for x in elements})
    stm = {
        "chunks": ["a", "b", "c", "d", "e"],
        "process": list(range(100, 600, 100)),
        "decay": list(range(200, 700, 100)),
    }
    for i in range(2):
        cipal.learn_chunks(ltm, stm, 1000)
    recode = cipal.find_chunks(stm["chunks"], ltm)
    stm = cipal.compress_stm(recode, stm, ltm, 1000)
    assert stm["chunks"] == ["a", "b c", "d e"]
    assert len(ltm) == 9


# Test that learning from recoded material leads to a BCDE chunk
def test_learn_chunks_bcde():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e"]
    ltm.update({x: 200 for x in elements})
    stm = {
        "chunks": ["a", "b", "c", "d", "e"],
        "process": list(range(100, 600, 100)),
        "decay": list(range(200, 700, 100)),
    }
    for i in range(2):
        cipal.learn_chunks(ltm, stm, 1000)
    recode = cipal.find_chunks(stm["chunks"], ltm)
    stm = cipal.compress_stm(recode, stm, ltm, 1000)
    cipal.learn_chunks(ltm, stm, 2000)
    assert "b c d e" in ltm
    assert len(ltm) == 10


# Test that learning again from recoded material leads to a ABCDE chunk
def test_learn_chunks_abcde():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e"]
    ltm.update({x: 200 for x in elements})
    stm = {
        "chunks": ["a", "b", "c", "d", "e"],
        "process": list(range(100, 600, 100)),
        "decay": list(range(200, 700, 100)),
    }
    for i in range(2):
        cipal.learn_chunks(ltm, stm, 1000)
    recode = cipal.find_chunks(stm["chunks"], ltm)
    stm = cipal.compress_stm(recode, stm, ltm, 1000)
    cipal.learn_chunks(ltm, stm, 2000)
    recode = cipal.find_chunks(stm["chunks"], ltm)
    stm = cipal.compress_stm(recode, stm, ltm, 1000)
    assert stm["chunks"] == ["a", "b c d e"]
    cipal.learn_chunks(ltm, stm, 2000)
    assert "a b c d e" in ltm
    assert len(ltm) == 11


# Test that CIPAL skips unprocessed parts to learn the processed sequences
def test_learn_chunks_skip_c():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e"]
    ltm.update({x: 200 for x in elements})
    stm = {
        "chunks": ["a", "b", "c", "d", "e"],
        "process": [100, 100, 5000, 100, 100],
        "decay": list(range(200, 700, 100)),
    }
    cipal.learn_chunks(ltm, stm, 1000)
    assert "a b" in ltm
    assert "d e" in ltm
    assert len(ltm) == 7


# Test that CIPAL learns BC first since D is unprocessed
def test_learn_chunks_bc():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e"]
    ltm.update({x: 200 for x in elements})
    stm = {
        "chunks": ["a", "b", "c", "d", "e"],
        "process": [100, 100, 100, 5000, 100],
        "decay": list(range(200, 700, 100)),
    }
    cipal.learn_chunks(ltm, stm, 1000)
    assert "b c" in ltm
    assert len(ltm) == 6


# Test that a second presentation leads to an AB chunk
def test_learn_chunks_ab():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e"]
    ltm.update({x: 200 for x in elements})
    stm = {
        "chunks": ["a", "b", "c", "d", "e"],
        "process": [100, 100, 100, 5000, 100],
        "decay": list(range(200, 700, 100)),
    }
    for i in range(2):
        cipal.learn_chunks(ltm, stm, 1000)
    assert "b c" in ltm
    assert "a b" in ltm
    assert len(ltm) == 7


# Test that BC is still used to recode the input
def test_learn_chunks_bc_recode():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e"]
    ltm.update({x: 200 for x in elements})
    stm = {
        "chunks": ["a", "b", "c", "d", "e"],
        "process": [100, 100, 100, 5000, 100],
        "decay": list(range(200, 700, 100)),
    }
    for i in range(2):
        cipal.learn_chunks(ltm, stm, 1000)
    recode = cipal.find_chunks(stm["chunks"], ltm)
    stm = cipal.compress_stm(recode, stm, ltm, 1000)
    assert stm["chunks"] == ["a", "b c", "d", "e"]
    assert len(ltm) == 7


# Test that learning from the recoded input leads to an ABC chunk
def test_learn_chunks_abc():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e"]
    ltm.update({x: 200 for x in elements})
    stm = {
        "chunks": ["a", "b", "c", "d", "e"],
        "process": [100, 100, 100, 5000, 100],
        "decay": list(range(200, 700, 100)),
    }
    for i in range(2):
        cipal.learn_chunks(ltm, stm, 1000)
    recode = cipal.find_chunks(stm["chunks"], ltm)
    stm = cipal.compress_stm(recode, stm, ltm, 1000)
    cipal.learn_chunks(ltm, stm, 2000)
    assert "a b c" in ltm
    assert len(ltm) == 8


# Test that CIPAL does not learn larger chunks since D is unprocessed
def test_learn_chunks_abc_max():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e"]
    ltm.update({x: 200 for x in elements})
    stm = {
        "chunks": ["a", "b", "c", "d", "e"],
        "process": [100, 100, 100, 5000, 100],
        "decay": list(range(200, 700, 100)),
    }
    for i in range(2):
        cipal.learn_chunks(ltm, stm, 1000)
    for i in range(2):
        recode = cipal.find_chunks(stm["chunks"], ltm)
        stm = cipal.compress_stm(recode, stm, ltm, 1000)
        cipal.learn_chunks(ltm, stm, 2000)
    assert len(ltm) == 8
    assert "a b c d" not in ltm
    assert "d e" not in ltm


# Test that the pt of a new chunk is the mean of the two subchunks
def test_learn_chunks_pt():
    ltm = cipal.new_ltm()
    stm = {"chunks": ["a", "b"], "process": [10, 10], "decay": [20, 20]}
    ltm["a"] = 100
    ltm["b"] = 300
    cipal.learn_chunks(ltm, stm, 1000)
    assert ltm["a b"] == 200


# find_chunks & compress_stm -----------------------------------------------------------


# Test that CIPAL returns the elements
def test_find_compress_elements():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e", "f"]
    ltm.update({x: 100 for x in elements})
    stm = {
        "chunks": ["a", "b", "c", "d", "e", "f"],
        "process": [200] * 6,
        "decay": [200] * 6,
    }
    recode = cipal.find_chunks(stm["chunks"], ltm)
    assert recode == list(range(6, 0, -1))
    stm = cipal.compress_stm(recode, stm, ltm, 100)
    assert stm["chunks"] == elements
    assert all(x == 200 for x in stm["process"])
    assert all(x == 200 for x in stm["decay"])
    assert len(stm["chunks"]) == len(stm["process"]) == len(stm["decay"]) == 6


# Test that CIPAL returns AB+CD+EF chunks
def test_find_compress_bigrams():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e", "f"]
    bigrams = ["a b", "c d", "e f"]
    ltm.update({x: 100 for x in elements})
    ltm.update({x: 100 for x in bigrams})
    stm = {
        "chunks": ["a", "b", "c", "d", "e", "f"],
        "process": [200] * 6,
        "decay": [200] * 6,
    }
    recode = cipal.find_chunks(stm["chunks"], ltm)
    assert recode == [3, 3, 2, 2, 1, 1]
    stm = cipal.compress_stm(recode, stm, ltm, 100)
    assert stm["chunks"] == bigrams
    assert all(x == 200 for x in stm["process"])
    assert all(x == 200 for x in stm["decay"])
    assert len(stm["chunks"]) == len(stm["process"]) == len(stm["decay"]) == 3


# Test that CIPAL returns ABC+DEF chunks
def test_find_compress_trigrams():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e", "f"]
    bigrams = ["a b", "c d", "e f"]
    trigrams = ["a b c", "d e f"]
    ltm.update({x: 100 for x in elements})
    ltm.update({x: 100 for x in bigrams})
    ltm.update({x: 100 for x in trigrams})
    stm = {
        "chunks": ["a", "b", "c", "d", "e", "f"],
        "process": [200] * 6,
        "decay": [200] * 6,
    }
    recode = cipal.find_chunks(stm["chunks"], ltm)
    assert recode == [2, 2, 2, 1, 1, 1]
    stm = cipal.compress_stm(recode, stm, ltm, 100)
    assert stm["chunks"] == trigrams
    assert all(x == 200 for x in stm["process"])
    assert all(x == 200 for x in stm["decay"])
    assert len(stm["chunks"]) == len(stm["process"]) == len(stm["decay"]) == 2


# Test that CIPAL returns ABCD+EF chunks
def test_find_compress_abcd():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e", "f"]
    bigrams = ["a b", "c d", "e f"]
    trigrams = ["a b c", "d e f"]
    ltm.update({x: 100 for x in elements})
    ltm.update({x: 100 for x in bigrams})
    ltm.update({x: 100 for x in trigrams})
    ltm["a b c d"] = 100
    stm = {
        "chunks": ["a", "b", "c", "d", "e", "f"],
        "process": [200] * 6,
        "decay": [200] * 6,
    }
    recode = cipal.find_chunks(stm["chunks"], ltm)
    assert recode == [1, 1, 1, 1, 2, 2]
    stm = cipal.compress_stm(recode, stm, ltm, 100)
    assert stm["chunks"] == ["a b c d", "e f"]
    assert all(x == 200 for x in stm["process"])
    assert all(x == 200 for x in stm["decay"])
    assert len(stm["chunks"]) == len(stm["process"]) == len(stm["decay"]) == 2


# Test that CIPAL returns AB+CDEF chunks
def test_find_compress_cdef():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e", "f"]
    bigrams = ["a b", "c d", "e f"]
    trigrams = ["a b c", "d e f"]
    ltm.update({x: 100 for x in elements})
    ltm.update({x: 100 for x in bigrams})
    ltm.update({x: 100 for x in trigrams})
    ltm["a b c d"] = 100
    ltm["c d e f"] = 100
    stm = {
        "chunks": ["a", "b", "c", "d", "e", "f"],
        "process": [200] * 6,
        "decay": [200] * 6,
    }
    recode = cipal.find_chunks(stm["chunks"], ltm)
    assert recode == [2, 2, 1, 1, 1, 1]
    stm = cipal.compress_stm(recode, stm, ltm, 100)
    assert stm["chunks"] == ["a b", "c d e f"]
    assert all(x == 200 for x in stm["process"])
    assert all(x == 200 for x in stm["decay"])
    assert len(stm["chunks"]) == len(stm["process"]) == len(stm["decay"]) == 2


# Test that CIPAL returns ABCDE+F chunks
def test_find_compress_abcde():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e", "f"]
    bigrams = ["a b", "c d", "e f"]
    trigrams = ["a b c", "d e f"]
    ltm.update({x: 100 for x in elements})
    ltm.update({x: 100 for x in bigrams})
    ltm.update({x: 100 for x in trigrams})
    ltm["a b c d"] = 100
    ltm["c d e f"] = 100
    ltm["a b c d e"] = 100
    stm = {
        "chunks": ["a", "b", "c", "d", "e", "f"],
        "process": [200] * 6,
        "decay": [200] * 6,
    }
    recode = cipal.find_chunks(stm["chunks"], ltm)
    assert recode == [1, 1, 1, 1, 1, 2]
    stm = cipal.compress_stm(recode, stm, ltm, 100)
    assert stm["chunks"] == ["a b c d e", "f"]
    assert all(x == 200 for x in stm["process"])
    assert all(x == 200 for x in stm["decay"])
    assert len(stm["chunks"]) == len(stm["process"]) == len(stm["decay"]) == 2


# Test that CIPAL returns A+BCDEF chunks
def test_find_compress_bcdef():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e", "f"]
    bigrams = ["a b", "c d", "e f"]
    trigrams = ["a b c", "d e f"]
    ltm.update({x: 100 for x in elements})
    ltm.update({x: 100 for x in bigrams})
    ltm.update({x: 100 for x in trigrams})
    ltm["a b c d"] = 100
    ltm["c d e f"] = 100
    ltm["a b c d e"] = 100
    ltm["b c d e f"] = 100
    stm = {
        "chunks": ["a", "b", "c", "d", "e", "f"],
        "process": [200] * 6,
        "decay": [200] * 6,
    }
    recode = cipal.find_chunks(stm["chunks"], ltm)
    assert recode == [2, 1, 1, 1, 1, 1]
    stm = cipal.compress_stm(recode, stm, ltm, 100)
    assert stm["chunks"] == ["a", "b c d e f"]
    assert all(x == 200 for x in stm["process"])
    assert all(x == 200 for x in stm["decay"])
    assert len(stm["chunks"]) == len(stm["process"]) == len(stm["decay"]) == 2


# Test that CIPAL returns ABCDEF chunks
def test_find_compress_abcdef():
    ltm = cipal.new_ltm()
    elements = ["a", "b", "c", "d", "e", "f"]
    bigrams = ["a b", "c d", "e f"]
    trigrams = ["a b c", "d e f"]
    ltm.update({x: 100 for x in elements})
    ltm.update({x: 100 for x in bigrams})
    ltm.update({x: 100 for x in trigrams})
    ltm["a b c d"] = 100
    ltm["c d e f"] = 100
    ltm["a b c d e"] = 100
    ltm["b c d e f"] = 100
    ltm["a b c d e f"] = 100
    stm = {
        "chunks": ["a", "b", "c", "d", "e", "f"],
        "process": [200] * 6,
        "decay": [200] * 6,
    }
    recode = cipal.find_chunks(stm["chunks"], ltm)
    assert recode == [1, 1, 1, 1, 1, 1]
    stm = cipal.compress_stm(recode, stm, ltm, 100)
    assert stm["chunks"] == ["a b c d e f"]
    assert all(x == 200 for x in stm["process"])
    assert all(x == 200 for x in stm["decay"])
    assert len(stm["chunks"]) == len(stm["process"]) == len(stm["decay"]) == 1


# decay_stm ----------------------------------------------------------------------------


# Test that decayed sequences are removed from STM
def test_decay_stm():
    stm = {
        "chunks": ["a", "b", "c", "d", "e"],
        "process": list(range(100, 600, 100)),
        "decay": list(range(100, 600, 100)),
    }
    stm = cipal.decay_stm(stm, 50)
    assert stm["chunks"] == ["a", "b", "c", "d", "e"]
    assert len(stm["chunks"]) == len(stm["process"]) == len(stm["decay"]) == 5
    stm = cipal.decay_stm(stm, 100)  # Chunks decay when time == decay (not > decay)
    assert stm["chunks"] == ["b", "c", "d", "e"]
    assert len(stm["chunks"]) == len(stm["process"]) == len(stm["decay"]) == 4
    stm = cipal.decay_stm(stm, 200)
    assert stm["chunks"] == ["c", "d", "e"]
    assert len(stm["chunks"]) == len(stm["process"]) == len(stm["decay"]) == 3
    stm = cipal.decay_stm(stm, 350)
    assert stm["chunks"] == ["d", "e"]
    assert len(stm["chunks"]) == len(stm["process"]) == len(stm["decay"]) == 2
    stm = cipal.decay_stm(stm, 400)
    assert stm["chunks"] == ["e"]
    assert len(stm["chunks"]) == len(stm["process"]) == len(stm["decay"]) == 1
    stm = cipal.decay_stm(stm, 550)
    assert stm["chunks"] == []
    assert len(stm["chunks"]) == len(stm["process"]) == len(stm["decay"]) == 0


# pt_sigmoid ---------------------------------------------------------------------------


# Test that pt_sigmoid generates non-linear adjustments to the chunk speeds
def test_pt_sigmoid():
    adjust = [x / 10.0 for x in range(1, 101, 1)]
    mid = list(range(200, 2050, 50))
    for a in adjust:
        for m in mid:
            # Basic test of the sigmoid curve for adjusting pt
            assert cipal.pt_sigmoid(m, m) == (0.8 / 2) + 0.2

            # Slowest chunks increase by ~99% of pt_adjust
            assert (cipal.pt_sigmoid(m * 2, m) * a) > (a * 0.99)

            # Fastest chunks increase by ~21% of pt_adjust
            assert (cipal.pt_sigmoid(m * 0.01, m) * a) < (a * 0.21)

            # Chunks at the inflection point increase by 60% of pt_adjust
            assert (round(cipal.pt_sigmoid(m, m), 10) * a) == (a * 0.6)

            # Larger adjustments are made for slower chunks
            assert (
                cipal.pt_sigmoid(m * 2, m)
                > cipal.pt_sigmoid(m * 1.8, m)
                > cipal.pt_sigmoid(m * 1.6, m)
                > cipal.pt_sigmoid(m * 1.4, m)
                > cipal.pt_sigmoid(m * 1.2, m)
                > cipal.pt_sigmoid(m, m)
                > cipal.pt_sigmoid(m * 0.8, m)
                > cipal.pt_sigmoid(m * 0.6, m)
                > cipal.pt_sigmoid(m * 0.4, m)
                > cipal.pt_sigmoid(m * 0.2, m)
                > cipal.pt_sigmoid(m * 0.1, m)
                > cipal.pt_sigmoid(m * 0.01, m)
                > cipal.pt_sigmoid(m * 0.001, m)
            )

            # Changes in adjustment magnitude are nonlinear
            assert (
                (cipal.pt_sigmoid(m * 0.2, m) - cipal.pt_sigmoid(m * 0.1, m))
                < (cipal.pt_sigmoid(m * 0.6, m) - cipal.pt_sigmoid(m * 0.5, m))
                < (cipal.pt_sigmoid(m * 1.2, m) - cipal.pt_sigmoid(m * 1.1, m))
                > (cipal.pt_sigmoid(m * 1.6, m) - cipal.pt_sigmoid(m * 1.5, m))
                > (cipal.pt_sigmoid(m * 2.2, m) - cipal.pt_sigmoid(m * 2.1, m))
            )

            # Similar adjustments are made either side of the inflection point
            assert round(
                cipal.pt_sigmoid(m * 1.2, m) - cipal.pt_sigmoid(m * 1.1, m), 10
            ) == round(cipal.pt_sigmoid(m * 0.9, m) - cipal.pt_sigmoid(m * 0.8, m), 10)
            assert round(
                cipal.pt_sigmoid(m * 1.5, m) - cipal.pt_sigmoid(m * 1.4, m), 10
            ) == round(cipal.pt_sigmoid(m * 0.6, m) - cipal.pt_sigmoid(m * 0.5, m), 10)
            assert round(
                cipal.pt_sigmoid(m * 1.9, m) - cipal.pt_sigmoid(m * 1.8, m), 10
            ) == round(cipal.pt_sigmoid(m * 0.2, m) - cipal.pt_sigmoid(m * 0.1, m), 10)


# adjust_pt ----------------------------------------------------------------------------


# Test that only A, B, and C get faster
def test_adjust_pt_positive():
    pt_initial = 1200
    pt_ceiling = 10
    pt_adjust = 10  # Positive adjustment parameter
    elements = ["a", "b", "c", "d", "e"]
    ltm = cipal.new_ltm()
    ltm.update({x: 600 for x in elements})
    stm = {"chunks": ["a", "b", "c"], "process": [200] * 3, "decay": [200] * 3}
    cipal.adjust_pt(ltm, stm, pt_adjust, pt_initial, pt_ceiling)
    assert ltm["a"] == 594
    assert ltm["b"] == 594
    assert ltm["c"] == 594
    assert ltm["d"] == 600
    assert ltm["e"] == 600


# Test that the results are the same with a negative parameter
def test_adjust_pt_negative():
    pt_initial = 1200
    pt_ceiling = 10
    pt_adjust = -10  # Negative adjustment parameter
    elements = ["a", "b", "c", "d", "e"]
    ltm = cipal.new_ltm()
    ltm.update({x: 600 for x in elements})
    stm = {"chunks": ["a", "b", "c"], "process": [200] * 3, "decay": [200] * 3}
    cipal.adjust_pt(ltm, stm, pt_adjust, pt_initial, pt_ceiling)
    assert ltm["a"] == 594
    assert ltm["b"] == 594
    assert ltm["c"] == 594
    assert ltm["d"] == 600
    assert ltm["e"] == 600


# Test that chunk speeds never exceed the ceiling level
def test_adjust_pt_ceiling():
    pt_initial = 1200
    pt_ceiling = 590
    pt_adjust = -50
    elements = ["a", "b", "c", "d", "e"]
    ltm = cipal.new_ltm()
    ltm.update({x: 600 for x in elements})
    stm = {
        "chunks": ["a", "b", "c", "d", "e"],
        "process": [200] * 5,
        "decay": [200] * 5,
    }
    cipal.adjust_pt(ltm, stm, pt_adjust, pt_initial, pt_ceiling)
    temp = cipal.ltm_to_df(ltm)
    assert all(x == 590 for x in temp["pt"])
    pt_ceiling = 700
    cipal.adjust_pt(ltm, stm, pt_adjust, pt_initial, pt_ceiling)
    temp = cipal.ltm_to_df(ltm)
    assert all(x == 700 for x in temp["pt"])
    pt_ceiling = 1000
    cipal.adjust_pt(ltm, stm, pt_adjust, pt_initial, pt_ceiling)
    temp = cipal.ltm_to_df(ltm)
    assert all(x == 1000 for x in temp["pt"])


# ltm_to_df ----------------------------------------------------------------------------


# Test that ltm_to_df pairs each chunk with the correct speeds
def test_ltm_to_df():
    ltm = cipal.new_ltm()
    ltm["a"] = 100
    ltm["b"] = 207
    ltm["c"] = 459
    temp = cipal.ltm_to_df(ltm)
    pt = temp["pt"][temp["chunks"] == "a"].values
    assert len(pt) == 1 and pt[0] == 100
    pt = temp["pt"][temp["chunks"] == "b"].values
    assert len(pt) == 1 and pt[0] == 207
    pt = temp["pt"][temp["chunks"] == "c"].values
    assert len(pt) == 1 and pt[0] == 459
    assert all(temp["chunks"] == ["a", "b", "c"])
    assert all(temp["pt"] == [100, 207, 459])
