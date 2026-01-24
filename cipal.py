"""

Chunk-based Incremental Processing and Learning (CIPAL)
Authors: Andrew Jessop, Julian Pine, and Fernand Gobet

"""

from math import exp

import pandas as pd


def learn(
    corpus,
    ltm,
    speech_rate=160,
    decay_rate=800,
    pt_adjust=5.0,
    pt_initial=1200.0,
    pt_ceiling=10.0,
):
    for utt in corpus:
        stream = utt.split()
        speech_times = list(
            range(0, (len(stream) * speech_rate) + decay_rate, speech_rate)
        )
        stm = new_stm()
        for i, t in enumerate(speech_times):
            if i < len(stream):
                learn_element(stream[i], ltm, pt_initial)
                add_to_stm(stream[i], stm, ltm, t, decay_rate)
            if len(stm["chunks"]) > 1:
                learn_chunks(ltm, stm, t)
                recode = find_chunks(stm["chunks"], ltm)
                stm = compress_stm(recode, stm, ltm, t)
            stm = decay_stm(stm, t)
            adjust_pt(ltm, stm, pt_adjust, pt_initial, pt_ceiling)


def process(items, ltm):
    # Raise an error if the items contain any unknown elements
    elements = set(element for item in items for element in item.split())
    unknown = [element for element in elements if element not in ltm]
    if unknown:
        raise ValueError(f"Items contain unknown elements: {unknown}")
    # Recode each item using the chunks stored in LTM
    parse_list, chunk_list, pt_list = [], [], []
    for item in items:
        stream = item.split()
        recode = find_chunks(stream, ltm)
        unique_indices = list(dict.fromkeys(recode))
        parse, pt = [], []
        for index in unique_indices:
            chunk_indices = [j for j, idx in enumerate(recode) if idx == index]
            chunk = " ".join([stream[j] for j in chunk_indices])
            parse.append(f"[{chunk}]")
            pt.append(ltm[chunk])
        parse_list.append(" ".join(parse))
        chunk_list.append(len(parse))
        pt_list.append(sum(pt))
    return pd.DataFrame(
        {"item": items, "parse": parse_list, "chunks": chunk_list, "pt": pt_list}
    )


def new_ltm():
    return {}


def new_stm():
    return {"chunks": [], "process": [], "decay": []}


def check_stm(stm):
    if not (len(stm["chunks"]) == len(stm["process"]) == len(stm["decay"])):
        raise ValueError("STM fields have different lengths.")


def learn_element(element, ltm, pt_initial):
    ltm.setdefault(element, pt_initial)


def add_to_stm(element, stm, ltm, time_t, decay_rate):
    stm["chunks"].append(element)
    stm["process"].append(float(ltm[element] + time_t))
    stm["decay"].append(float(decay_rate + time_t))
    check_stm(stm)


def learn_chunks(ltm, stm, time_t):
    chunks_rev = stm["chunks"][::-1]
    process_rev = stm["process"][::-1]
    unused = [True] * len(chunks_rev)
    for j in range(1, len(chunks_rev)):
        if (
            unused[j]
            and unused[j - 1]
            and (time_t >= process_rev[j])
            and (time_t >= process_rev[j - 1])
        ):
            new_c = f"{chunks_rev[j]} {chunks_rev[j - 1]}"
            if new_c not in ltm:
                ltm[new_c] = (ltm[chunks_rev[j]] + ltm[chunks_rev[j - 1]]) / 2
                unused[j] = unused[j - 1] = False


def find_chunks(stm_chunks, ltm):
    recode = [0] * len(stm_chunks)
    start_index, end_index, chunk_id, adjust = 0, len(stm_chunks), 1, 0
    while 0 in recode:
        # Define the sequence
        sequence = " ".join(stm_chunks[start_index:end_index])
        # Use the same chunks to parse the input if the sequence contains only one chunk
        if (end_index - start_index == 1) and recode[start_index] == 0:
            recode[start_index] = chunk_id
            chunk_id += 1
            start_index -= 1
            end_index -= 1
        # If a longer sequence is coded, check if it is recognized
        elif sequence in ltm.keys() and all(
            [j == 0 for j in recode[start_index:end_index]]
        ):
            recode[start_index:end_index] = [chunk_id] * (end_index - start_index)
            chunk_id += 1
        # If it is not recognized, reduce the size of the sequence
        elif start_index == 0:
            adjust += 1
            start_index += adjust
            end_index = len(stm_chunks)
        # Or slide the recode window back to find an alternative chunk
        else:
            start_index -= 1
            end_index -= 1
    return recode


def compress_stm(recode, stm, ltm, time_t):
    stm_recode = {"chunks": [], "process": [], "decay": []}
    unique_indices = list(dict.fromkeys(recode))  # Preserve the original index order
    for index in unique_indices:
        chunk_indices = [j for j, idx in enumerate(recode) if idx == index]
        chunk = " ".join([stm["chunks"][j] for j in chunk_indices])
        stm_recode["chunks"].append(chunk)
        # Make no changes to the timings that have not been recoded
        if len(chunk_indices) == 1:
            stm_recode["process"].append(stm["process"][chunk_indices[0]])
            stm_recode["decay"].append(stm["decay"][chunk_indices[0]])
        # Set the chunk decay time to the most recent chunk in the recode
        else:
            stm_recode["process"].append(ltm[chunk] + time_t)
            stm_recode["decay"].append(stm["decay"][chunk_indices[-1]])
    check_stm(stm_recode)
    return stm_recode


def decay_stm(stm, time_t):
    active_chunks = [time_t < decay_time for decay_time in stm["decay"]]
    stm_active = {
        "chunks": [
            chunk for chunk, active in zip(stm["chunks"], active_chunks) if active
        ],
        "process": [
            process for process, active in zip(stm["process"], active_chunks) if active
        ],
        "decay": [
            decay for decay, active in zip(stm["decay"], active_chunks) if active
        ],
    }
    check_stm(stm_active)
    return stm_active


def pt_sigmoid(pt, mid):
    return (0.8 / (1 + exp((mid - pt) / (mid * 0.2)))) + 0.2


def adjust_pt(ltm, stm, pt_adjust, pt_initial, pt_ceiling):
    for chunk in stm["chunks"]:
        ltm[chunk] = max(
            ltm[chunk] + (-abs(pt_adjust) * pt_sigmoid(ltm[chunk], pt_initial / 2)),
            pt_ceiling,
        )


def ltm_to_df(ltm):
    return pd.DataFrame(list(ltm.items()), columns=["chunks", "pt"])
