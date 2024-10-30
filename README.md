# Enhancing Chess Reinforcement Learning with Graph Representation

# How to Run

## Step 1: Create a venv and download the requirements

`requirements.txt` installs the cuda version of jaxlib by default, which
requires a GPU and cuda.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 1.5: Activate the venv

```bash
. venv/bin/activate
```

## Step 2: Run the example

By default, an AIM server is expected to run on port 53800. It is possible to
run AIM on a remote server, and to use an SSH port tunnel to make it available
locally with the command `ssh -N -L 53800:localhost:53800 <remote server>`.

Running

```bash
python train.py
```

will start to train a full model. The run can be configured by editing the
`config` dictionary in `train.py`. If the 75% of allocated GPU memory is an
issue, it is possible to pass `XLA_PYTHON_CLIENT_ALLOCATOR=platform` as an
environment variable.

```bash
python ranking.py <model>
```

will select 5 opponents for the model run `<model>`, and run matches. Those
matches are saved as pgn in `tournaments/ranking matches`, the outcomes
are saved in `rankings.json`, and estimated ELO ratings are evaluated and
stored for each player.

# Book explorer

The file `book.py` can be used to explore all the saved pgn games. For
example:

```bash
python book.py tournaments/ranking\ matches/ -p="chess_2024-08-20:00h13-499" -o="e2e4 c7c5 Nb1c3 e7e6 Ng1f3 Nb8c6 d2d4 c5d4 Nf3d4 Ng8f6 Nd4c6 b7c6 e4e5 Nf6d5 Nc3e4 Qd8c7 f2f4 Qc7b6"
```

Will return one of the games included in the appendix of the paper.
