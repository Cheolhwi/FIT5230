import argparse, os, time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipelineSafe
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig

def main():
    parser = argparse.ArgumentParser(description="Safe Stable Diffusion quick runner")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--safety", type=str, default="medium", choices=["weak", "medium", "strong", "max"],
                        help="Safety configuration preset")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipelineSafe.from_pretrained("AIML-TUDA/stable-diffusion-safe").to(device)

    # Map safety preset to SafetyConfig kwargs
    preset_map = {
        "weak": SafetyConfig.WEAK,
        "medium": SafetyConfig.MEDIUM,
        "strong": SafetyConfig.STRONG,
        "max": SafetyConfig.MAX,
    }
    safety_kwargs = preset_map[args.safety]

    gen = torch.Generator(device=device).manual_seed(args.seed)

    out = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=gen,
        **safety_kwargs,
    )
    img = out.images[0]

    Path("outputs").mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = Path("outputs") / f"sld_{args.safety}_{ts}.png"
    img.save(out_path)
    print(f"Saved: {out_path.resolve()}")
    print("Active safety concept:", pipe.safety_concept)

if __name__ == "__main__":
    main()
