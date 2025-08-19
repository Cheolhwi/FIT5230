# Safe Stable Diffusion (Colab Starter)

This repository contains a Colab-friendly starter to run **Safe Stable Diffusion** using the built-in
`StableDiffusionPipelineSafe` from the 🤗 diffusers library. It includes a ready-made notebook and a small CLI script.

## Quick start (Colab)
1. Open the notebook directly in Colab (public repos):  
   `https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/notebooks/updated_safe_latent_diffusion_sample.ipynb`
2. **Runtime → Change runtime type → GPU (T4 or better)**.
3. Run the first cell to install dependencies.
4. Log into Hugging Face (`from huggingface_hub import login; login()`), then accept model terms on the model card if prompted.
5. Execute the rest of the cells to generate images.

## Local quick start (optional, CPU/GPU)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
# If you plan to run the script:
# huggingface-cli login
python run_safe_sd.py --prompt "a serene lake at sunrise, ultra-detailed" --seed 1234 --safety strong
```

Images will be saved under `outputs/` (ignored by git).

## Notes
- The code uses `StableDiffusionPipelineSafe` and the checkpoint `"AIML-TUDA/stable-diffusion-safe"`.
- You might need to accept the model license on its Hugging Face page before the first run.
- Colab usually comes with a suitable PyTorch build; we intentionally do **not** pin `torch` in `requirements.txt`.
- This repo only contains **code**. The underlying model is licensed under **CreativeML OpenRAIL-M**; please review the model card.
