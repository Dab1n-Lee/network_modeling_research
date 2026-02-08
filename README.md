## Reproducibility

```bash
git clone --recurse-submodules https://github.com/Dab1n-Lee/network_modeling_research.git
conda env create -f environment.yml
conda activate audio_research

python scripts/01_make_stimuli.py
python scripts/02_aextract_audiomae_embeddings.py
python scripts/03b_pitch_geometry_audiomae_from_npz.py
```