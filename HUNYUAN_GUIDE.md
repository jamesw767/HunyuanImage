# HunyuanImage 3.0 Complete Guide

This guide covers prompting techniques, model architecture, and tips for getting the best results from Tencent's HunyuanImage-3.0.

## Why Is This Model So Large?

HunyuanImage-3.0 has **80 billion total parameters** (with 13B activated per token during inference), making it the largest open-source image generation model. Here's why:

### Mixture-of-Experts (MoE) Architecture
- 64 expert networks collaborate, with only ~13B parameters activated per token
- This balances massive model capacity with computational efficiency
- Experts become increasingly specialized for different modalities (text vs image)
- One shared MLP plus 8 experts activated per token

### Native Multimodal Design
Unlike traditional diffusion models (Stable Diffusion, FLUX) that use separate text encoders (CLIP) bolted onto image generators:
- HunyuanImage-3.0 uses a **unified autoregressive framework**
- Text and image understanding happen in the same backbone
- Enables world-knowledge reasoning and better semantic understanding
- No pipeline switching between understanding and generation

### Training Scale
- Trained on **5 billion high-quality images** (curated from 10B raw images)
- Hierarchical captioning from 30 to 1,000+ words per image
- Post-training with DPO, GRPO, and proprietary ReDA optimization

## Key Advantages Over Other Models

| Feature | HunyuanImage 3.0 | FLUX | Stable Diffusion XL |
|---------|------------------|------|---------------------|
| Token Limit | **512+ tokens (mT5)** | ~77 (CLIP) | ~77 (CLIP) |
| Multi-element Accuracy | **91%** | 78% | 72% |
| Text Rendering | **Excellent (Chinese & English)** | Good | Poor |
| Long Prompt Support | **1000+ characters** | Limited | Limited |
| Reasoning | **Chain-of-thought capable** | None | None |

### What Makes It Special

1. **World Knowledge Reasoning**: Can interpret sparse prompts and fill in contextually appropriate details
2. **Superior Text Rendering**: Accurately renders text in images (both English and Chinese)
3. **Complex Scene Composition**: Handles 8-10 distinct elements with proper spatial relationships
4. **Extended Prompts**: Process paragraph-length descriptions without truncation

---

## Prompting Techniques

### How HunyuanImage 3.0 Differs from SD/FLUX

**Traditional models (SD, FLUX):**
- Use CLIP text encoder limited to ~77 tokens
- Respond well to comma-separated keyword lists
- Support bracket emphasis syntax: `(important:1.4)`, `((very important))`
- Separate negative prompt field

**HunyuanImage 3.0:**
- Uses mT5 encoder supporting **512+ tokens**
- Prefers **prose-style descriptions** over keyword lists
- **Does NOT use bracket/weight syntax** - it's an autoregressive LLM, not diffusion
- Negative concepts embedded in prompt with "no" keywords (e.g., "no watermark")

### Recommended Prompt Structure

HunyuanImage-3.0 works best with a **hierarchical prompt framework**:

```
[Main Subject] + [Subject Details] + [Environment/Setting] +
[Composition & Framing] + [Lighting & Atmosphere] +
[Artistic Style] + [Technical Specifications] + [Mood/Emotion]
```

#### Example - Good Prompt:
```
A weathered lighthouse keeper in his 60s with a salt-and-pepper beard,
wearing a thick navy wool sweater, standing at the top of a spiral
staircase inside a Victorian lighthouse. Late afternoon golden hour
light streaming through the lantern room windows, creating dramatic
rim lighting. Shot from below looking up, emphasizing the spiral
architecture. Photorealistic style with cinematic color grading,
shallow depth of field, 35mm film grain. Atmosphere of solitude
and timeless dedication.
```

#### Example - What NOT to Do:
```
lighthouse keeper, old man, beard, sweater, lighthouse, beautiful,
amazing, stunning, dramatic, cinematic, 8k, masterpiece
```

### The 5-Step Refinement Loop

1. **Anchor**: Core objective in one crisp sentence at the top
2. **Constrain**: Add 2-3 hard constraints (lens, lighting, composition)
3. **Clarify**: Replace vague adjectives with measurable cues ("f/2.0", "85mm", "overcast diffuse")
4. **Compare**: Generate 4 variations, keep 1-2 best, note what worked
5. **Lock Style**: Save successful prompts as templates

### Text Rendering in Images

HunyuanImage-3.0 excels at rendering text. Use quotation marks for exact text:

```
A vintage coffee shop storefront with the text "ARTISAN COFFEE"
in bold serif font on a hand-painted wooden sign above the door
```

**Tips:**
- Put exact text in quotation marks
- Specify font style (serif, sans-serif, script, bold, italic)
- Describe placement and material (painted on wood, neon sign, carved stone)

### Multi-Subject Relationships

When including multiple subjects, **explicitly define spatial relationships**:

```
A young girl sitting on her father's shoulders, her hands resting
on his head, both looking at fireworks in the night sky
```

NOT:
```
girl, father, fireworks, night sky
```

### Avoiding Common Problems

| Problem | Solution |
|---------|----------|
| Unwanted text/logos | Add "no text, no logo, no watermark" |
| Extra limbs | Add "anatomically correct, no extra limbs" |
| Blurry results | Specify "sharp focus, high detail, crisp edges" |
| Wrong composition | Use camera terms: "centered composition", "rule of thirds" |
| Contradictory output | Avoid mixing conflicting atmospheres in one prompt |

### Cinematic Vocabulary

The model responds well to specific lighting and photography terms:

**Lighting:**
- Rembrandt lighting, rim lighting, chiaroscuro
- Golden hour, blue hour, overcast diffuse
- Harsh shadows, soft shadows, ambient occlusion

**Camera:**
- 35mm, 50mm, 85mm lens
- f/1.4 (shallow DOF), f/8 (deep focus)
- Low angle, high angle, Dutch angle
- Wide shot, medium shot, close-up

**Materials:**
- "brushed aluminum with directional grain"
- "frosted glass with soft diffusion"
- "weathered oak with visible grain"

---

## Generation Parameters

### Guidance Scale (CFG)

**What it does:** Controls how closely the image follows your prompt

| Range | Effect |
|-------|--------|
| 1.0-3.0 | More creative/abstract, model takes liberties |
| 4.0-7.0 | **Balanced** (default: 5.0) |
| 8.0-15.0 | Strict prompt adherence, may cause artifacts |

**Tips:**
- Start with 5.0 (default)
- Increase for precise text rendering or specific compositions
- Decrease for more artistic/unexpected results

### Inference Steps

| Steps | Speed | Quality |
|-------|-------|---------|
| 10-15 | Fast | Draft quality |
| 20 | **Standard** | Good balance |
| 30-50 | Slow | Maximum detail |

More steps = finer detail but diminishing returns past ~30

### Image Sizes

The model supports various aspect ratios:
- 1:1 (1024x1024) - Square, social media
- 16:9 (1024x576) - Landscape, cinematic
- 9:16 (576x1024) - Portrait, mobile
- 4:3, 3:2, 21:9 - Various use cases

The model can auto-select optimal size based on prompt content.

---

## Negative Prompts

HunyuanImage-3.0 handles negative concepts differently than diffusion models:

### Embed "No" Keywords in Your Prompt
```
A professional headshot photograph, clean background,
no watermark, no text, no logo, sharp focus
```

### Common Negative Keywords
```
no watermark, no text, no logo, no signature, no border,
no frame, no extra limbs, no deformed hands,
anatomically correct, clean edges, no artifacts
```

### The UI's Negative Prompt Field
When you use the negative prompt field in the UI, it appends as:
```
[your prompt]. Avoid: [negative prompt]
```

This works but embedded "no" keywords often work better for this model.

---

## Prompt Enhancement with Ollama

The UI can use local LLMs (Ollama) to enhance simple prompts into detailed ones.

### Length Settings
| Setting | Words | Best For |
|---------|-------|----------|
| minimal | 15-30 | Quick tests |
| short | 30-60 | Simple subjects |
| **medium** | 60-100 | General use |
| long | 100-150 | Complex scenes |
| detailed | 150-250 | Maximum detail |

### Complexity Settings
| Setting | Detail Level |
|---------|--------------|
| simple | Subject only |
| basic | Subject + setting |
| **detailed** | Full scene description |
| complex | Cinematic detail + mood |

### Recommended: DeepSeek for Enhancement
The official recommendation from Tencent is to use DeepSeek models for prompt rewriting, as the pretrain checkpoint doesn't auto-enhance prompts.

---

## Wildcard System

This UI supports wildcard syntax for randomized prompts:

### Basic Wildcards
```
A [animal] in a [landscape] setting
→ "A tiger in a forest setting"
```

### Combined Wildcards (+ syntax)
```
A [color+animal] in [mood+landscape]
→ "A golden dragon in a mysterious forest"
```

### Available Categories
Run `python wildcard_utils.py` to see all categories, or check `wildcards.json`

---

## Best Practices Summary

1. **Write prose, not keywords** - Full sentences work better than comma lists
2. **Be specific** - "85mm lens at f/2.8" beats "professional photo"
3. **Define relationships** - Explicitly state how subjects interact
4. **Use quotation marks for text** - Required for accurate text rendering
5. **Layer your description** - Subject → Details → Environment → Light → Style
6. **Start with default CFG (5.0)** - Adjust only if needed
7. **Embed negatives in prompt** - "no watermark" works better than negative prompt field
8. **Use prompt enhancement** - Ollama can transform simple ideas into detailed prompts
9. **Save successful prompts** - Build a template library for consistency
10. **Iterate one variable at a time** - Change only one thing between tests

---

## References

- [Official GitHub Repository](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)
- [HunyuanImage 3.0 Technical Report](https://arxiv.org/html/2509.23951v1)
- [Hugging Face Model Card](https://huggingface.co/tencent/HunyuanImage-3.0)
- [Advanced Prompt Engineering Guide](https://yuanic.com/blog/hunyuan-image-3-advanced-prompt-engineering-guide)
- [Complete Beginner's Guide](https://yuanic.com/blog/how-to-use-hunyuan-image-beginners-guide)
- [FAQ & Tips](https://yuanic.com/blog/hunyuan-image-3-faq)
- [Model Comparison Guide](https://dev.to/czmilo/tencent-hunyuan-image-30-complete-guide-in-depth-analysis-of-the-worlds-largest-open-source-57k3)
