# Graphs and MCTS with EdgeSets

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

Running

```bash
python train.py
```

will start to train a full model. The run can be configured by editing the
`config` dictionary in `train.py`.

```bash
python ranking.py
```

will select 5 opponents for each players defined in lines 216-218, and run
matches. The outcomes of those matches are saved in `rankings.json`, and
estimated ELO ratings are evaluated and stored for each player.
