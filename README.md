# CIPAL: Chunk-based Incremental Processing and Learning

This repository contains the latest implementation of CIPAL in [Python](https://www.python.org/). It is intended for researchers looking for the most up-to-date version of CIPAL. You are welcome to use any materials in this repository in accordance with the *CC-BY Attribution 4.0 International* license. If you run simulations with CIPAL in your research, please cite the article below and not this repository directly:

**Jessop, A., Pine, J., & Gobet, F. (2025). Chunk-based incremental processing and learning: An integrated theory of word discovery, implicit statistical learning, and speed of lexical processing. Psychological Review, 132(6), 1340–1374. <https://doi.org/10.1037/rev0000564>**

We also recommend that you include a copy of the CIPAL code used in any published work as supplementary materials (e.g., in an OSF repository).

All project dependencies are managed using [uv](https://docs.astral.sh/uv/).

## Switching from Julia to Python

All future updates to CIPAL will be implemented in Python and uploaded to this repository.

The original version of CIPAL presented in [Jessop et al. (2025)](https://doi.org/10.1037/rev0000564) was implemented in [Julia](https://julialang.org). We decided to use Julia since it offers a familiar high-level syntax (similar to [*R*](https://www.r-project.org/) or [Python](https://www.python.org/)) for writing [fast and efficient code](https://julialang.org/benchmarks/). However, Julia remains a niche language focused on numeric and scientific computing, with a small community of users and developers. In comparison, Python is one of the most widely used programming languages (see the latest [TIOBE index](https://www.tiobe.com/tiobe-index/)), with a large community working on improving the language and developing specialist packages for modelling and data analysis. Recent updates to Python have also led to significant performance improvements, making it possible to run simulations quickly in CIPAL, even with large datasets (see Benchmarks below). Therefore, we decided to port CIPAL to Python to make the code more transparent and accessible, allowing more researchers to use, scrutinize, adapt, or extend the architecture. We hope the contents of this repository will help others to understand CIPAL and the theory-driven testing methodology.

## Changes to CIPAL since version 1.0.0

The code for CIPAL v1.0.0 is included in the [supplemental materials](https://osf.io/fhrxg/) of [Jessop et al. (2025)](https://doi.org/10.1037/rev0000564). In addition to porting the codebase to Python, we have redesigned the model to improve its performance, transparency, and maintainability. However, the core behaviour of CIPAL is identical to v1.0.0 (as shown in `2_process.ipynb` and `3_canonical.ipynb`). All changes are listed below:

- LTM is now implemented as a hash table (i.e., a Dictionary in Python) rather than as lists. This has drastically improved the runtime for models trained with large quantities of corpus data (see benchmarks below).
- The CIPAL module is now organised into one script for the source code (`cipal.py`), one script for the unit tests (`1_unit_test.py`), and Jupyter notebooks for the process tests (`2_process.ipynb`) and canonical results tests (`3_canonical.ipynb`). We have found that storing the functions together in one script makes it easier for others to download and navigate the source code.
- We renamed the processing time variables; they are now prefixed with "pt" rather than "sop" (e.g., `sop_used` → `pt_used`).
- We removed the option to reduce the processing times of unused chunks from the `learn` function, since chunk decay is not part of the core CIPAL theory. This simplifies the code, as any (non-zero) value supplied to the`pt_all` parameter will be used to reduce the speed of all chunks in LTM with each processing cycle. In CIPAL v1.0.0, negative values were used to reduce processing times, whereas positive values produced an increase (i.e., chunk decay).

## Benchmarks

To assess the performance of CIPAL in Python, we measured the amount of time it took to train the model with different quantities of child-directed speech: 10,000, 100,000, and 1,000,000 utterances. For each input level, we generated 10 random samples of the [English-NA](https://talkbank.org/childes/access/Eng-NA/) collection from [CHILDES](https://talkbank.org/childes/). To provide context for these run-times, we ran identical simulations using an identical Julia version of CIPAL v1.1.0 (with the changes listed above). All simulations were run on the same Apple MacBook Pro (16-inch, November 2023, Apple M3 Max, 48 GB, macOS Tahoe 26.2), providing an "apples-to-apples" comparison. The table below shows the average run-time across the 10 samples:

| *N* utterances | Julia (secs) | Python (secs) |
| :------------- |:------------:|:----------:   |
| 10,000         | 0.32         |0.32           |
| 100,000        | 2.44         |4.58           |
| 1,000,000      | 21.9         |43.9           |


## Theory-driven testing

The CIPAL architecture was designed and built according to the theory-driven testing methodology (see [Lane & Gobet, 2012](https://doi.org/10.1080/0952813X.2012.695443)). As well as the code for the architecture itself (`cipal.py`), this repository contains a set of automated unit tests (`1_unit_test.py`), process tests (`2_process.ipynb`), and canonical results tests (`3_canonical.ipynb`). The unit tests where written with the [pytest](https://docs.pytest.org/en/stable/) package.

Before running any models with `cipal.py`, you should check that the source code for the architecture works correctly on your system. All the tests in the `1_unit_test.py` script should pass, and the results in each Jupyter notebook should match those in the corresponding `.html` files.


## Repository contents

- `pyproject.toml` - Project information and dependencies
- `uv.lock` - [uv](https://docs.astral.sh/uv/) generated lock file
- `cipal.py` – The latest version of the CIPAL codebase
- `1_unit_test.py` – Set of automated unit tests for CIPAL
- `2_process.ipynb` (and `.html`) – Jupyter notebook for the process tests
- `3_canonical.ipynb`(and `.html`) – Jupyter notebook for the canonical results tests
- `data.zip` – Data for running the automated tests
    - `cds` – Data for the child-directed speech simulations
        - `corpus.txt` – English NA collection from CHILDES
        - `cdi.txt` – MB-CDI word list (English USA)
    - `tasks` – Data for four statistical learning tasks
        - `exposure` – 100 random exposures per study
        - `items` – Test items for each study


## How to use CIPAL

Below is an example of a simple simulation where CIPAL is trained with three different utterances and then tested with the different word pairs that appear within these utterances. For converting orthographic text into phonemes, we used [eSpeak NG](https://github.com/espeak-ng/espeak-ng). For these symbols to render correctly in Windows, unicode UTF-8 must be enabled in the language settings.

```python
import cipal
```

```python
# Repeat each utterance 30 times
input = [
    "D e@ I t w 0 z", # "there it was"
    "w 0 z I t D e@", # "was it there"
    "I t w 0 z D e@" # "it was there"
] * 30
```

```python
ltm = cipal.new_ltm()
cipal.learn(input, ltm)
```

```python
cipal.ltm_to_df(ltm)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chunks</th>
      <th>pt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>D</td>
      <td>456.160784</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e@</td>
      <td>609.526916</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I</td>
      <td>456.160784</td>
    </tr>
    <tr>
      <th>3</th>
      <td>t</td>
      <td>609.526916</td>
    </tr>
    <tr>
      <th>4</th>
      <td>w</td>
      <td>462.011346</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>625.314462</td>
    </tr>
    <tr>
      <th>6</th>
      <td>z</td>
      <td>609.526916</td>
    </tr>
    <tr>
      <th>7</th>
      <td>w 0</td>
      <td>453.276763</td>
    </tr>
    <tr>
      <th>8</th>
      <td>z I</td>
      <td>620.512437</td>
    </tr>
    <tr>
      <th>9</th>
      <td>t D</td>
      <td>620.512437</td>
    </tr>
    <tr>
      <th>10</th>
      <td>I t</td>
      <td>334.672217</td>
    </tr>
    <tr>
      <th>11</th>
      <td>w 0 z</td>
      <td>290.888436</td>
    </tr>
    <tr>
      <th>12</th>
      <td>D e@</td>
      <td>380.275786</td>
    </tr>
    <tr>
      <th>13</th>
      <td>w 0 z D e@</td>
      <td>372.013308</td>
    </tr>
    <tr>
      <th>14</th>
      <td>D e@ I t</td>
      <td>340.294628</td>
    </tr>
    <tr>
      <th>15</th>
      <td>w 0 z I t</td>
      <td>381.276291</td>
    </tr>
    <tr>
      <th>16</th>
      <td>w 0 z I t D e@</td>
      <td>315.263698</td>
    </tr>
    <tr>
      <th>17</th>
      <td>I t w 0 z</td>
      <td>315.057566</td>
    </tr>
    <tr>
      <th>18</th>
      <td>D e@ I t w 0 z</td>
      <td>293.470437</td>
    </tr>
    <tr>
      <th>19</th>
      <td>I t w 0 z D e@</td>
      <td>319.911382</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Create a list of bigrams for testing the model
bigrams = ["I t w 0 z", "I t D e@", "w 0 z D e@", "D e@ I t", "w 0 z I t"]
```

```python
# Process the test items without learning
cipal.process(bigrams, ltm)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item</th>
      <th>parse</th>
      <th>chunks</th>
      <th>pt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I t w 0 z</td>
      <td>[I t w 0 z]</td>
      <td>1</td>
      <td>315.057566</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I t D e@</td>
      <td>[I t] [D e@]</td>
      <td>2</td>
      <td>714.948004</td>
    </tr>
    <tr>
      <th>2</th>
      <td>w 0 z D e@</td>
      <td>[w 0 z D e@]</td>
      <td>1</td>
      <td>372.013308</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D e@ I t</td>
      <td>[D e@ I t]</td>
      <td>1</td>
      <td>340.294628</td>
    </tr>
    <tr>
      <th>4</th>
      <td>w 0 z I t</td>
      <td>[w 0 z I t]</td>
      <td>1</td>
      <td>381.276291</td>
    </tr>
  </tbody>
</table>
</div>


## Funding

The development of the CIPAL architecture was funded by the [ESRC LuCiD Centre](https://lucid.ac.uk/) at the [University of Liverpool](https://www.liverpool.ac.uk/) (ES/S007113/1 and ES/L008955/1). We have no conflicts of interest to disclose.
