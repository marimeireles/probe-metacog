"""
Model loading, hook utilities, and generation for Gemma 3 IT models (4B/27B).

Key corrections from original plan:
- Uses Gemma3ForConditionalGeneration (the actual cached model)
- Layers accessed via model.language_model.layers[i]
- Head patching hooks o_proj INPUT (pre-projection)
  because o_proj input dim != hidden_size (can't reverse this reshape)
- Response extraction: decode only generated tokens, not the prompt
- 27B model uses device_map="auto" across multiple GPUs
"""

import os
import torch

# Disable torch dynamo to avoid cache limit issues during repeated generation
os.environ["TORCHDYNAMO_DISABLE"] = "1"
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

from typing import Optional, Callable
import config as cfg

# ── Sampling kwargs (built once, reused by all generate calls) ────────
_sampling_kwargs = {}
if cfg.DO_SAMPLE:
    _sampling_kwargs = dict(temperature=cfg.TEMPERATURE, top_k=50, top_p=0.95)


def load_model_and_tokenizer():
    """Load Gemma 3 IT model and tokenizer from local snapshot path.

    Uses the direct filesystem path to avoid HuggingFace Hub resolution,
    which fails on compute nodes without internet access.
    Supports both 4B (single GPU) and 27B (multi-GPU with device_map="auto").
    """
    os.environ["HF_HOME"] = cfg.HF_CACHE
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    model_path = cfg.MODEL_LOCAL_PATH
    print(f"Loading model: {cfg.MODEL_ID} (size={cfg.MODEL_SIZE})", flush=True)
    print(f"  Path: {model_path}", flush=True)
    print(f"  Temperature: {cfg.TEMPERATURE}  do_sample: {cfg.DO_SAMPLE}  results_dir: {cfg.RESULTS_DIR}", flush=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    print(f"  Loaded on devices: {set(str(p.device) for p in model.parameters())}", flush=True)
    return model, tokenizer


def get_text_layers(model):
    """Get the text model's decoder layers.

    Structure: Gemma3ForConditionalGeneration
      .model (Gemma3Model)
        .language_model (Gemma3TextModel) ← has .layers directly
    """
    return model.model.language_model.layers


def get_text_model(model):
    """Get the inner text model (Gemma3TextModel)."""
    return model.model.language_model


# ── Layer output helpers ──────────────────────────────────────────────
# transformers v5+ decoder layers return hidden_states tensor directly;
# older versions returned a tuple (hidden_states, ...).

def _get_hidden_states(out):
    """Extract hidden_states from a decoder layer's forward output."""
    if isinstance(out, tuple):
        return out[0]
    return out


def _set_hidden_states(out, modified):
    """Return a modified layer output with replaced hidden_states."""
    if isinstance(out, tuple):
        return (modified,) + out[1:]
    return modified


# ── Residual stream hooks ─────────────────────────────────────────────

def extract_residual(model, tokenizer, text, layer_idx):
    """
    Extract the residual stream activation at the LAST token position
    from a specific layer.

    Returns: tensor of shape [1, hidden_size] = [1, 2560]
    """
    acts = {}
    layers = get_text_layers(model)

    def hook(mod, inp, out, idx=layer_idx):
        acts[idx] = _get_hidden_states(out).detach().clone()

    h = layers[layer_idx].register_forward_hook(hook)
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    h.remove()

    # Return last token activation: [1, hidden_size]
    return acts[layer_idx][:, -1, :]


def extract_residual_all_positions(model, tokenizer, text, layer_idx):
    """
    Extract the residual stream activation at ALL token positions.

    Returns: tensor of shape [1, seq_len, hidden_size]
    """
    acts = {}
    layers = get_text_layers(model)

    def hook(mod, inp, out, idx=layer_idx):
        acts[idx] = _get_hidden_states(out).detach().clone()

    h = layers[layer_idx].register_forward_hook(hook)
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    h.remove()

    return acts[layer_idx]


# ── Concept vector computation ────────────────────────────────────────

def compute_concept_vector(model, tokenizer, word, all_words, layer_idx):
    """
    Compute a concept direction vector using mean-difference method.

    target = activation for "Tell me about {word}."
    baseline = mean activation for "Tell me about {w}." for all other words
    concept_vec = normalize(target - baseline)

    Returns: tensor of shape [1, 2560], unit-normalized
    """
    target = extract_residual(
        model, tokenizer, f"Tell me about {word}.", layer_idx
    )

    baselines = []
    for w in all_words:
        if w != word:
            b = extract_residual(
                model, tokenizer, f"Tell me about {w}.", layer_idx
            )
            baselines.append(b)

    baseline = torch.stack(baselines).mean(dim=0)
    vec = target - baseline
    vec = vec / vec.norm()
    return vec  # [1, 2560]


# ── Injection calibration ─────────────────────────────────────────────

def calibrate_injection_strengths(model, tokenizer, layer_idx, fractions):
    """
    Convert relative strength fractions to absolute strengths by measuring
    the residual stream L2 norm at a given layer.

    Gemma 3 4B uses sqrt(hidden_size) embedding scaling, giving residual norms
    ~31000. Absolute strengths of 1-16 (as in Lindsey) are < 0.1% perturbation.

    Args:
        fractions: list of floats (e.g., [0.01, 0.05, 0.1])
    Returns:
        absolute_strengths: list of floats, and the residual_norm
    """
    # Measure residual norm with a neutral prompt
    residual = extract_residual(model, tokenizer, "Tell me about the weather.", layer_idx)
    norm = residual.norm().item()
    absolute_strengths = [frac * norm for frac in fractions]
    return absolute_strengths, norm


# ── Injection hooks ───────────────────────────────────────────────────

def make_injection_hook(concept_vec, strength):
    """
    Create a hook that adds a scaled concept vector to the residual stream.

    The hook adds to hidden_states at all token positions.
    During autoregressive generation with KV cache, only the current
    token position is processed, so the injection applies to each new token.

    Args:
        concept_vec: [1, 2560] unit-normalized concept direction
        strength: scalar multiplier
    Returns:
        hook function
    """
    scaled = (concept_vec * strength).detach()

    def hook(module, inp, out):
        hidden_states = _get_hidden_states(out)
        # scaled is [1, 2560], broadcast to [B, S, 2560]
        modified = hidden_states + scaled.unsqueeze(1).to(hidden_states.device, hidden_states.dtype)
        return _set_hidden_states(out, modified)

    return hook


def _decode_generated_only(tokenizer, full_output_ids, input_length):
    """Decode only the generated tokens (after the input prompt)."""
    generated_ids = full_output_ids[input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_with_injection(model, tokenizer, input_ids, attention_mask,
                            concept_vec, layer_idx, strength,
                            max_tokens=cfg.MAX_NEW_TOKENS):
    """
    Generate text with concept vector injection at a specific layer.

    Returns: decoded string (ONLY the generated response, not the prompt)
    """
    layers = get_text_layers(model)
    hook_fn = make_injection_hook(concept_vec, strength)
    h = layers[layer_idx].register_forward_hook(hook_fn)

    input_len = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=cfg.DO_SAMPLE,
            **_sampling_kwargs,
        )

    h.remove()
    return _decode_generated_only(tokenizer, out[0], input_len)


def generate_plain(model, tokenizer, input_ids, attention_mask,
                   max_tokens=cfg.MAX_NEW_TOKENS):
    """Generate text without any injection. Returns only the response."""
    input_len = input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=cfg.DO_SAMPLE,
            **_sampling_kwargs,
        )
    return _decode_generated_only(tokenizer, out[0], input_len)


# ── Chat template helpers ─────────────────────────────────────────────

def build_chat_input(tokenizer, messages, add_generation_prompt=True):
    """
    Build tokenized input from a list of message dicts.

    Args:
        messages: list of {"role": "user"/"model"/"system", "content": str}
        add_generation_prompt: whether to add the model turn prefix

    Returns:
        dict with input_ids and attention_mask tensors
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    inputs = tokenizer(text, return_tensors="pt")
    return inputs


def build_exp1_input(tokenizer):
    """Build Experiment 1 input (system + user prompt for thought detection)."""
    messages = [
        {"role": "system", "content": cfg.EXP1_SYSTEM},
        {"role": "user", "content": cfg.EXP1_USER},
    ]
    try:
        return build_chat_input(tokenizer, messages)
    except Exception:
        # Fallback: embed system prompt in user message
        messages = [
            {"role": "user",
             "content": cfg.EXP1_SYSTEM + "\n\n" + cfg.EXP1_USER},
        ]
        return build_chat_input(tokenizer, messages)


# ── Activation recording during generation ────────────────────────────

def generate_and_record_activations(model, tokenizer, input_ids, attention_mask,
                                    layer_idx, max_tokens=cfg.MAX_NEW_TOKENS):
    """
    Generate text AND record the residual stream activations at each
    generated token position (for Experiment 4 cosine similarity analysis).

    Returns:
        text: decoded output string (response only)
        activations: list of tensors, each [1, 2560], one per generated token
    """
    layers = get_text_layers(model)
    token_acts = []

    def record_hook(mod, inp, out):
        # During generation, hidden_states is [B, 1, 2560] (single token)
        token_acts.append(_get_hidden_states(out)[:, -1, :].detach().clone())

    h = layers[layer_idx].register_forward_hook(record_hook)

    input_len = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=cfg.DO_SAMPLE,
            **_sampling_kwargs,
        )

    h.remove()

    text = _decode_generated_only(tokenizer, out[0], input_len)
    # The first activation in token_acts corresponds to the prompt processing
    # (all prompt tokens at once). Skip it - we only want generated token activations.
    generated_acts = token_acts[1:] if len(token_acts) > 1 else token_acts
    return text, generated_acts


# ── Head-level patching ───────────────────────────────────────────────

def make_head_patch_hook(head_idx, replacement_activations):
    """
    Create a hook on o_proj that patches a specific head's output
    BEFORE the output projection.

    The input to o_proj has shape [B, S, 2048] = [B, S, 8 * 256].
    We reshape to [B, S, 8, 256], replace head_idx, reshape back.

    Args:
        head_idx: which of the 8 query heads to patch (0-7)
        replacement_activations: tensor to substitute for that head
    """
    def hook(module, args):
        # forward_pre_hook: args is a tuple of inputs to o_proj
        x = args[0]  # [B, S, 2048]
        B, S, _ = x.shape
        # Reshape to per-head: [B, S, 8, 256]
        x = x.view(B, S, cfg.NUM_QUERY_HEADS, cfg.HEAD_DIM)
        # Patch the specific head
        repl = replacement_activations.to(x.device, x.dtype)
        if repl.shape[1] != S:
            # If sequence lengths differ, patch what we can (last S positions)
            repl = repl[:, -S:, :]
        x[:, :, head_idx, :] = repl
        # Reshape back: [B, S, 2048]
        x = x.view(B, S, -1)
        return (x,) + args[1:]

    return hook


def get_head_activations(model, tokenizer, text, layer_idx, head_idx):
    """
    Extract a specific head's activations (pre-o_proj) during a forward pass.

    Returns: tensor of shape [1, seq_len, head_dim] = [1, S, 256]
    """
    head_acts = {}
    layers = get_text_layers(model)

    def hook(module, args):
        x = args[0]  # [B, S, 2048]
        B, S, _ = x.shape
        x_heads = x.view(B, S, cfg.NUM_QUERY_HEADS, cfg.HEAD_DIM)
        head_acts['val'] = x_heads[:, :, head_idx, :].detach().clone()
        return args  # don't modify

    h = layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(hook)
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    h.remove()

    return head_acts['val']


def patch_head_and_generate(model, tokenizer, input_ids, attention_mask,
                            concept_vec, layer_idx, head_idx, strength,
                            hit_head_activations, max_tokens=cfg.MAX_NEW_TOKENS):
    """
    Generate with:
    1. Concept injection at the layer level (same as normal injection)
    2. Head-level patching: replace one head's pre-o_proj activations

    Returns: decoded response string
    """
    layers = get_text_layers(model)

    # Layer-level injection hook
    inject_hook_fn = make_injection_hook(concept_vec, strength)
    inject_h = layers[layer_idx].register_forward_hook(inject_hook_fn)

    # Head-level patch hook (on o_proj forward_pre_hook)
    patch_hook_fn = make_head_patch_hook(head_idx, hit_head_activations)
    patch_h = layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(patch_hook_fn)

    input_len = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=cfg.DO_SAMPLE,
            **_sampling_kwargs,
        )

    inject_h.remove()
    patch_h.remove()

    return _decode_generated_only(tokenizer, out[0], input_len)


# ── Cosine similarity utility ─────────────────────────────────────────

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = a.float().flatten()
    b = b.float().flatten().to(a.device)
    return torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)


# ── Cached forward passes (for SAE / attribution analysis) ──────────

def forward_with_cache(model, tokenizer, text, layers_to_cache=None):
    """Run a single forward pass caching layer output activations.

    Args:
        text: input string
        layers_to_cache: list of layer indices, or None for all layers

    Returns:
        cache: dict {layer_idx: tensor [1, seq_len, hidden_size]} (detached)
        logits: output logits tensor [1, seq_len, vocab_size] (detached)
    """
    layers = get_text_layers(model)
    if layers_to_cache is None:
        layers_to_cache = list(range(len(layers)))

    cache = {}
    hooks = []

    for idx in layers_to_cache:
        def hook(mod, inp, out, layer_idx=idx):
            cache[layer_idx] = _get_hidden_states(out).detach().clone()
        hooks.append(layers[idx].register_forward_hook(hook))

    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    for h in hooks:
        h.remove()

    logits = outputs.logits.detach() if hasattr(outputs, 'logits') else None
    return cache, logits


def forward_with_cache_grad(model, tokenizer, text, layers_to_cache=None):
    """Run a forward pass caching activations WITH gradients (not detached).

    For attribution patching: activations retain grad_fn so we can
    compute gradients w.r.t. them via torch.autograd.grad().

    Args:
        text: input string
        layers_to_cache: list of layer indices, or None for all layers

    Returns:
        cache: dict {layer_idx: tensor [1, seq_len, hidden_size]} (WITH grad)
        logits: output logits tensor (WITH grad)
        input_ids: the tokenized input_ids tensor
    """
    layers = get_text_layers(model)
    if layers_to_cache is None:
        layers_to_cache = list(range(len(layers)))

    cache = {}
    hooks = []

    for idx in layers_to_cache:
        def hook(mod, inp, out, layer_idx=idx):
            hs = _get_hidden_states(out)
            # Clone but keep in graph via requires_grad.
            # Return as new output so computation flows through cached tensor,
            # enabling torch.autograd.grad(metric, cached) to work.
            cached = hs.clone()
            cached.requires_grad_(True)
            cached.retain_grad()
            cache[layer_idx] = cached
            return _set_hidden_states(out, cached)
        hooks.append(layers[idx].register_forward_hook(hook))

    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # No torch.no_grad — need gradients
    outputs = model(**inputs)

    for h in hooks:
        h.remove()

    logits = outputs.logits if hasattr(outputs, 'logits') else None
    return cache, logits, inputs["input_ids"]


def forward_with_injection_and_cache(model, tokenizer, text, concept_vec,
                                      inject_layer_idx, strength,
                                      layers_to_cache=None):
    """Forward pass with injection at one layer, caching activations at others.

    Args:
        text: input string
        concept_vec: [1, hidden_size] unit-normalized concept direction
        inject_layer_idx: layer to inject at
        strength: scalar multiplier for injection
        layers_to_cache: list of layer indices to cache (default: same as inject)

    Returns:
        cache: dict {layer_idx: tensor [1, seq_len, hidden_size]} (detached)
        logits: output logits tensor (detached)
    """
    layers = get_text_layers(model)
    if layers_to_cache is None:
        layers_to_cache = [inject_layer_idx]

    cache = {}
    hooks = []

    # Injection hook
    inject_fn = make_injection_hook(concept_vec, strength)
    hooks.append(layers[inject_layer_idx].register_forward_hook(inject_fn))

    # Cache hooks
    for idx in layers_to_cache:
        def cache_hook(mod, inp, out, layer_idx=idx):
            cache[layer_idx] = _get_hidden_states(out).detach().clone()
        hooks.append(layers[idx].register_forward_hook(cache_hook))

    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    for h in hooks:
        h.remove()

    logits = outputs.logits.detach() if hasattr(outputs, 'logits') else None
    return cache, logits
