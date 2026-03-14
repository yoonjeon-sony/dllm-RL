
### Create Venv
```
uv venv
source .venv/bin/activate

uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
uv pip install xformers==0.0.31
uv pip install --no-build-isolation --no-cache-dir "flash-attn<=2.8.0"
uv sync
```

### Scripts

#### SFT Training
```
sbatch scripts/
```
- Download dataset from 
- Download SFT trained checkpoint from yjyjyj98/sft-lavidao-thinkmorph-edit

####  RL Training
```
sbatch scripts/
```

### Data Processing
```
python data/zebracot.py
```