# Code Repository

## Setup

On HPC node with Rocky Linux:

1. `module load Python/3.12.3`
2. `python -m venv venv`
3. `source venv/bin/activate`
4. `pip install -r requirements.txt`

## Dev mode

On local PC, open port forward with

1. `ssh -L 8888:localhost:8888 ab123456@login...htp.itc.rwth-aachen.de`

Then

2. `./dev.sh` in project dir
3. Open the link locally

