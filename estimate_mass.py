"""
Post-processing Layer 2: estimate object mass.

Design: a fallback chain, not a single model call.

  1. spec_weight_kg, if you already scraped it from the product page -- exact,
     always preferred when available.
  2. density_estimate: volume (now known exactly, thanks to Layer 1) x a
     material-density lookup, where material is classified by the VLM.
     Grounded in a real physical formula rather than pure pattern-matching.
  3. vlm_direct_estimate: ask the VLM for a single mass number directly,
     given the image + exact dimensions + category. Least reliable --
     used only if 1 and 2 aren't available/confident.

Mass estimation from an image is fundamentally under-determined (a full vs.
empty milk carton look identical), so every result carries a `source` and
`confidence` field -- don't drop that information downstream in the URDF
export step.

VLM backend: OpenAI-compatible chat-completions endpoint, so it works with
any open-weight model served through vLLM / Ollama / a hosted API using
that schema. Suggested open-source options (see chat writeup for tradeoffs):
  - Qwen2.5-VL / Qwen3-VL (7B-72B)   -- easiest to self-host, strong default
  - InternVL3 (8B-78B)               -- strong open alternative
  - Kimi-VL-A3B                      -- lightweight MoE, cheap to run
  - Kimi K3                          -- native vision, frontier-class, but
                                          2.8T params (16/896 experts active)
                                          -- realistically API-served, not
                                          self-hosted, for this subtask
"""
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("estimate_mass")

# Rough material density lookup (kg/m^3), deliberately coarse -- this is a
# fallback estimator, not a materials science tool. Extend as needed.
MATERIAL_DENSITY_KG_M3 = {
    "wood_solid": 700,
    "wood_particleboard": 650,
    "plastic": 950,
    "ceramic": 2300,
    "glass": 2500,
    "metal_steel": 7850,
    "metal_aluminum": 2700,
    "cardboard": 275,
    "fabric_foam": 100,
    "bread_baked_dough": 250,
    "unknown": 500,
}

MASS_PROMPT_TEMPLATE = """You are estimating the physical mass of a real object for a robotics \
simulation. You will not be perfectly accurate, and that's fine -- give your best single-number \
estimate, not a refusal.

Object image is attached. Known exact dimensions (from the product spec sheet): \
width={width_m:.3f}m, height={height_m:.3f}m, depth={depth_m:.3f}m (volume envelope \
{volume_m3:.5f} m^3, note real solid volume is usually smaller than the bounding box).
Category / name: {category}

Respond with ONLY a JSON object, no other text, no markdown fences:
{{
  "material": "<short material label, e.g. 'ceramic', 'solid wood', 'cardboard', 'baked dough'>",
  "estimated_mass_kg": <float>,
  "confidence": "<low|medium|high>",
  "reasoning": "<one short sentence>"
}}"""


@dataclass
class MassResult:
    mass_kg: float
    source: str          # "spec_sheet" | "density_estimate" | "vlm_direct"
    material: str
    confidence: str       # "exact" | "high" | "medium" | "low"
    reasoning: str = ""


def _encode_image_b64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _call_vlm(
    image_path: Path,
    prompt: str,
    base_url: str,
    model: str,
    api_key: str = "not-needed",
) -> dict:
    """
    Calls any OpenAI-compatible chat-completions VLM endpoint
    (vLLM / Ollama / Moonshot API / etc). Import is local so this module
    doesn't hard-require the `openai` package unless this path is used.
    """
    from urllib.parse import urlparse

    import httpx
    from openai import OpenAI  # pip install openai

    # On networks with a corporate HTTP(S)_PROXY set, httpx (which the
    # openai client uses) will route *localhost* requests through that
    # proxy too unless NO_PROXY explicitly excludes it -- the proxy then
    # can't reach our own loopback address and the call fails with a
    # Squid/whatever "connection refused" error page. Since a local vLLM/
    # Ollama endpoint should never go through an external proxy anyway,
    # bypass the environment proxy config outright for loopback hosts
    # rather than relying on NO_PROXY being set correctly in every shell.
    host = urlparse(base_url).hostname
    is_local = host in ("localhost", "127.0.0.1", "::1")
    http_client = httpx.Client(trust_env=not is_local)

    client = OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
    img_b64 = _encode_image_b64(image_path)
    ext = image_path.suffix.lstrip(".").lower() or "png"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext};base64,{img_b64}"},
                    },
                ],
            }
        ],
        temperature=0.2,
        max_tokens=300,
    )
    raw = response.choices[0].message.content.strip()
    # tolerate models that wrap JSON in markdown fences despite instructions
    raw = raw.strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    return json.loads(raw)


def estimate_mass(
    image_path: Path,
    width_m: float,
    height_m: float,
    depth_m: float,
    category: str = "unknown object",
    spec_weight_kg: Optional[float] = None,
    vlm_base_url: str = "http://localhost:8000/v1",
    vlm_model: str = "qwen2.5-vl-7b-instruct",
    vlm_api_key: str = "not-needed",
    solid_fill_ratio: float = 0.6,
) -> MassResult:
    """
    Fallback chain: spec sheet > density estimate > direct VLM guess.

    solid_fill_ratio: fraction of the bounding-box volume assumed to be
    actual solid material (objects are rarely solid cuboids -- a plate is
    mostly a thin shell, a loaf of bread has a rounded top). 0.6 is a
    generic placeholder; tune per category if you want better accuracy.
    """
    volume_m3 = width_m * height_m * depth_m

    # 1. spec sheet -- exact, always wins
    if spec_weight_kg is not None:
        return MassResult(
            mass_kg=spec_weight_kg,
            source="spec_sheet",
            material="unknown",
            confidence="exact",
            reasoning="Taken directly from product spec sheet.",
        )

    # 2 & 3: need the VLM to classify material (for density lookup) or to
    # give a direct guess. One call serves both purposes.
    prompt = MASS_PROMPT_TEMPLATE.format(
        width_m=width_m, height_m=height_m, depth_m=depth_m,
        volume_m3=volume_m3, category=category,
    )
    try:
        parsed = _call_vlm(image_path, prompt, vlm_base_url, vlm_model, vlm_api_key)
        material = str(parsed.get("material", "unknown")).lower().replace(" ", "_")
        vlm_mass = float(parsed.get("estimated_mass_kg", 0) or 0)
        confidence = str(parsed.get("confidence", "low")).lower()
        reasoning = str(parsed.get("reasoning", ""))
    except Exception as e:
        logger.warning(f"[estimate_mass] VLM call failed ({e}); falling back to generic density estimate.")
        material, vlm_mass, confidence, reasoning = "unknown", 0.0, "low", f"VLM call failed: {e}"

    # 2. density-based estimate, grounded in the now-exact volume -- prefer
    # this over the VLM's raw number when we have a recognized material,
    # since it's anchored in a physical formula rather than pure guessing.
    density = MATERIAL_DENSITY_KG_M3.get(material)
    if density is not None:
        density_mass_kg = volume_m3 * solid_fill_ratio * density
        return MassResult(
            mass_kg=round(density_mass_kg, 4),
            source="density_estimate",
            material=material,
            confidence="medium",
            reasoning=(
                f"volume({volume_m3:.5f}m^3) x fill_ratio({solid_fill_ratio}) x "
                f"density({density}kg/m^3) for material='{material}' (VLM-classified). {reasoning}"
            ),
        )

    # 3. last resort: trust the VLM's direct number
    if vlm_mass > 0:
        return MassResult(
            mass_kg=round(vlm_mass, 4),
            source="vlm_direct",
            material=material,
            confidence=confidence if confidence in ("low", "medium", "high") else "low",
            reasoning=reasoning,
        )

    # everything failed -- return an obviously-flagged placeholder rather
    # than silently writing 0 kg into a URDF (which breaks most physics
    # engines outright)
    logger.error("[estimate_mass] All estimation paths failed; returning placeholder mass.")
    return MassResult(
        mass_kg=0.5,
        source="placeholder",
        material="unknown",
        confidence="low",
        reasoning="All estimation methods failed; using arbitrary 0.5kg placeholder -- replace before use.",
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Estimate object mass via VLM + density fallback.")
    parser.add_argument("image_path", type=str)
    parser.add_argument("--width_m", type=float, required=True)
    parser.add_argument("--height_m", type=float, required=True)
    parser.add_argument("--depth_m", type=float, required=True)
    parser.add_argument("--category", type=str, default="unknown object")
    parser.add_argument("--spec_weight_kg", type=float, default=None)
    parser.add_argument("--vlm_base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--vlm_model", type=str, default="qwen2.5-vl-7b-instruct")
    parser.add_argument("--report_json", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = estimate_mass(
        image_path=Path(args.image_path),
        width_m=args.width_m, height_m=args.height_m, depth_m=args.depth_m,
        category=args.category, spec_weight_kg=args.spec_weight_kg,
        vlm_base_url=args.vlm_base_url, vlm_model=args.vlm_model,
    )
    print(json.dumps(asdict(result), indent=2))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()