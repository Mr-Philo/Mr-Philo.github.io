---
title: 'Megaton-LM Training Large Models Practical Guide | 2 - Model Construct'
date: 2026-02-04
permalink: /posts/2026/02/megatron-exp-2/
tags:
  - Megatron-LM
  - Practical Guide
header:
    teaser: /images/posts/2026-02-04-megatron-exp-2/cover.png
excerpt: "A practical guide to constructing and modifying GPT-style models in Megatron-LM: code organization, the Spec-based layer system, parameter flow, and how to switch between local and Transformer Engine implementations without getting lost."
---

![Blog Image](/images/posts/2026-02-04-megatron-exp-2/cover.png)

> This blog is also available in Chinese version: [https://zhuanlan.zhihu.com/p/1992607526580134947](https://zhuanlan.zhihu.com/p/1992607526580134947)

# Megaton-LM Training Large Models Practical Guide | 2 - Model Construct

Since the Transformer architecture was introduced, its popularity has never cooled down. Today there are countless codebases that teach you how to build your own Transformer from scratch, and many mature libraries that provide highly integrated and fast Transformer implementations. Among them, Megatron-LM is not necessarily the fastest—but it is very likely the most complex. Customizing model parameters inside Megatron-LM can be surprisingly difficult.

This post focuses on the Megatron-LM training framework and explains how to construct a trainable neural network from simple to complex. Because Megatron-LM’s structure is highly complex and somewhat redundant, there are many ways to construct models, and this seemingly simple step is actually full of pitfalls. From a research-engineering perspective, I will also describe how to modify model structure within the Megatron-LM codebase while trying to avoid common traps.

---
## 1. Code Organization Related to Model Construction

Before getting into details, we need to understand Megatron-LM’s (complicated) code organization. The project has gone through multiple iterations, so several parallel implementations coexist—one major reason people step on so many landmines.

### Core directory structure

```text
megatron/
├── core/                    # New core implementation (recommended)
│   ├── models/              # Implementations of various models
│   │   ├── gpt/             # GPT family
│   │   ├── mamba/           # Mamba models
│   │   └── multimodal/      # Multimodal models
│   └── transformer/         # Transformer building blocks
├── legacy/                  # Legacy implementation (being phased out)
│   └── model/               # Traditional model implementation
└── training/                # Training-related code
```

### Model construction entry points

Note: the function names and line numbers below are based on a Megatron-LM version around Jan 2026. As the project evolves, exact names and line numbers may change, but the overall structure usually remains similar.

Let’s start from the entry file for training a classic GPT model: `pretrain_gpt.py` in the Megatron-LM root directory. In that file, there are two model-construction-related functions: `model_provider` and `gpt_builder`. This API has been refactored: in versions before roughly Oct 2025, only `model_provider` existed; today, `model_provider` is essentially a wrapper around `gpt_builder`.

#### `model_provider.py` (root): a unified model provider interface

This is the model construction entry for the whole framework—a “universal model factory”. It accepts a model builder (e.g., `gpt_builder`, `mamba_builder`) and provides a consistent calling interface. Regardless of which model you build, you go through this entry. It can also handle some shared concerns such as memory monitoring and ModelOpt optimizations.

#### `model_provider.py` (root): the GPT-specific builder

This is the concrete GPT builder called by `model_provider`. It supports loading configuration from YAML or CLI args, choosing between the legacy and core systems, selecting an appropriate Transformer-layer implementation based on your requirements, and finally instantiating the model.

You can already see how complicated the build branching is. When defining the layer spec (note that Megatron’s *layer spec* and *GPT model spec* are not the same; they have separate logic), Megatron-LM provides multiple choices:

- If you specify a custom spec → use `args.spec`
- If MoE is enabled → use `get_gpt_decoder_block_spec()`
- If heterogeneous layer config is enabled → use `get_gpt_heterogeneous_layer_spec()`
- Otherwise → call `_get_transformer_layer_spec()`

`_get_transformer_layer_spec()` is a “layer spec selector” that decides which Transformer implementation to use. There are three main paths (discussed later):

- Transformer Engine path (`config.transformer_impl == "transformer_engine"`): high-performance GPU-optimized implementation (the main focus of this post)
- Inference-optimized path (`config.transformer_impl == "inference_optimized"`): optimized for inference (not discussed here)
- Local implementation path (`config.transformer_impl == "local"`): standard PyTorch implementation, useful for validating smaller model changes

⚠️ **Pitfall #1: choose the correct build path**

Megatron-LM has two parallel model construction systems:

- The new **core** system: `megatron/core/` (more complete)
- The old **legacy** system: `megatron/legacy/` (often more compatible)

The key switch is `--use-legacy-models`. If you modified code in the core system but launch training with legacy models, your changes will not take effect.

```python
# gpt_builders.py (example logic)
if args.use_legacy_models:
    model = megatron.legacy.model.GPTModel(...)  # legacy path
else:
    model = GPTModel(...)  # core path
```

Also note that the model is *built* inside `megatron/training/training.py`, but the actual model-definition functions live across other files. This often causes beginners to get lost when trying to modify model structure.

---
## 2. Constructing a Standard Transformer in Megatron-LM

Megatron-LM’s Transformer construction pipeline can be summarized in four major steps:

1. Argument parsing and configuration
2. Layer-spec definition
3. Model instantiation
4. Weight initialization

Let’s go through the key parts.

### 2.1 Key files and classes

Conceptually, a GPT model consists of:

- An embedding layer (e.g., `TextEmbedding`) that encodes input tokens
- A Transformer block (a stack of Transformer layers)
- An output layer (LM head) that maps hidden states to logits

The tricky part is that Megatron-LM spreads these parts across different files rather than defining everything in one place (unlike the HuggingFace Transformers style). So you have to read the code with more care.

#### 1) GPT model main class

- File: `megatron/core/models/gpt/gpt_model.py`
- Class: `GPTModel`
- Role: integrates embedding, Transformer block, and output head

```python
# GPTModel.__init__ (high-level sketch)
self.embedding = LanguageModelEmbedding(...)                 # token embedding
self.decoder = TransformerBlock(...)                        # main Transformer body
self.output_layer = tensor_parallel.ColumnParallelLinear(...)  # LM head
```

If you want to modify the embedding layer or output head, you generally do it here, or define a new `MyGPTModel` class. Most modifications fall into two categories:

- Initialization changes → modify `__init__()`
- Forward-pass changes → modify `forward()`

If you want to modify the backward pass… that’s much harder (and beyond the scope of this post). For small changes, PyTorch autograd often handles it. For more invasive changes, you may need to implement a custom `torch.autograd.Function` and expose it for Megatron to call.

#### 2) Transformer block

- File: `megatron/core/transformer/transformer_block.py` (note: different folder from the model file)
- Class: `TransformerBlock`
- Role: manages the stacking of multiple Transformer layers

If you want to change logic across layers, or build mixed stacks (e.g., Mamba-Transformer hybrids or MLP-MoE mixtures), this is a good place to look.

#### 3) Transformer layer

- File: `megatron/core/transformer/transformer_layer.py`
- Class: `TransformerLayer`
- Role: implements a single Transformer layer (attention + MLP)

If you want to change intra-layer computation (e.g., pre-norm vs post-norm ordering, attention/MLP wiring), this is usually where you do it.

Below that level (attention modules, MLP modules, etc.), the code moves into another set of abstractions. This leads to a core Megatron concept: **Spec-based layer definitions**.

---
### 2.2 Layer specs and modular design philosophy

To modify model structure in Megatron-LM, you need to understand its modular philosophy: the model is defined via a “Spec” system that allows flexible component replacement.

- File: `megatron/models/gpt/gpt_layer_specs.py`

If the GPT model / Transformer block / Transformer layer is the “skeleton”, then below that Megatron decomposes the layer into smaller modules (attention, layernorm, MLP, MoE, etc.)—like recipes for a big Transformer. `gpt_layer_specs.py` is essentially the “cookbook” that defines how to assemble different types of Transformer layers, and it is one of the most important files in the build system.

It has two core roles:

1) **A central kitchen for layer specs.** Each function is a “recipe”. The three main recipes correspond to different backends:

- `get_gpt_layer_with_transformer_engine_spec()`  # TE high-performance
- `get_gpt_layer_with_inference_spec()`           # inference optimized
- `get_gpt_layer_local_spec()`                    # standard PyTorch

2) **A backend abstraction system.** Different providers implement the same interface:

- `TESpecProvider`: NVIDIA Transformer Engine backend
- `LocalSpecProvider`: standard PyTorch ops
- `InferenceSpecProvider`: inference-optimized backend
- `KitchenSpecProvider`: NVIDIA Kitchen backend

A classic example of a local Transformer-layer spec:

```python
return ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=layer_norm,
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=backend.column_parallel_linear(),
                core_attention=backend.core_attention(),
                linear_proj=backend.row_parallel_linear(),
                q_layernorm=(
                    L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                ),
                k_layernorm=(
                    L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                ),
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=layer_norm,
        mlp=mlp,
        mlp_bda=get_bias_dropout_add,
        sharded_state_dict_keys_map={
            "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
            "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
        },
    ),
)
```

Whether you use TE or local implementation, whether you want MoE or standard Transformer, you define everything through the same interface. Once you are comfortable with this design, switching implementations becomes much easier:

- Change activation function → adjust `activation_func` in `MLPSubmodules`
- Swap linear implementations → replace `linear_qkv`, `linear_proj`, etc.
- Add a new attention mechanism → implement new `SelfAttentionSubmodules` and reference it here
- Implement heterogeneous layers → study the pattern used by `get_gpt_decoder_block_spec`

---
### 2.3 The complicated parameter flow

Megatron-LM exposes many CLI options for fine-grained control over model structure (see `megatron/training/argument.py`). The path from CLI args to final model configuration goes through multiple transformations:

- `arguments.py`: parse CLI args
- `core_transformer_config_from_args()`: convert to `TransformerConfig`
- `TransformerConfig.post_init()`: preprocess and validate config
- Module `__init__()` methods: apply parameters

Almost all model-structure-related args are eventually consolidated into a single `TransformerConfig` (defined in `megatron/core/transformer/transformer_config.py`). And this leads to another big pitfall.

⚠️ **Pitfall #2: multiple layers of implicit parameter transformation**

Some derived-argument behavior happens in `argument.py`; some happens during the conversion to `TransformerConfig`; some happens inside `TransformerConfig`’s validation; and some happens even later inside module initialization or forward.

For example, in `TransformerConfig.__post_init__`, if `ffn_hidden_size` is not set, it may default to $4\times\text{hidden\_size}`. But when using gated activations such as GLU or SwiGLU, `ffn_hidden_size` may effectively be doubled because the first part of the MLP contains two linear projections. In Megatron-LM, this adjustment is often not done in `TransformerConfig` validation but inside the MLP module initialization.

There are many similar implicit checks and “auto-fixes”. It is very hard to verify correctness by static code reading alone. I strongly recommend verifying end-to-end: build the model, then validate it, then go back to debug if something is off.

#### How to verify your model was built correctly

A common practical approach is to print the model structure and basic stats. Megatron-LM does not print these by default, so you may add something like:

```python
print(f"Model structure: {model}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Config: {model.config}")
```

---
## 3. Building Models with Transformer Engine (TE)

Transformer Engine (TE) is NVIDIA’s high-performance Transformer implementation. It can significantly accelerate training—especially if you want to leverage low-precision Tensor Core kernels on H/B-series GPUs (e.g., FP8/FP4). In such cases, using TE is highly recommended.

Megatron-LM’s TE support is fairly mature. The main control switch is:

- `--transformer-impl` (values: `transformer_engine` or `local`)

Relevant file and function:

- File: `megatron/core/extensions/transformer_engine_spec_provider.py`
- Function: `get_gpt_layer_with_transformer_engine_spec()`

A common workflow is to validate changes with the simpler local implementation, then switch to TE for performance. In that case, you should pay attention to differences between the two implementations.

### TE vs local: common component differences

Transformer Engine version:

```python
# fused high-performance operators
self_attention = TELayerNormColumnParallelLinear  # fused LayerNorm + Linear
mlp = TEFusedMLP                                   # fused MLP
input_layernorm = TENorm                            # TE LayerNorm
```

Local implementation version:

```python
# standard PyTorch operators
self_attention = ColumnParallelLinear              # standard Linear
mlp = MLP                                          # standard MLP
input_layernorm = FusedLayerNorm                   # standard LayerNorm
```

TE and local implementations can have compatibility issues: some APIs differ (Linear/LayerNorm/MLP, etc.), and checkpoint formats may differ, so checkpoint loading needs care. Also, some newer features might only exist in TE.

To confirm your model is using TE rather than local operators, you can do a simple check:

```python
for name, module in model.named_modules():
    if "attention" in name:
        print(f"{name}: {type(module)}")
        break
```

If you see TE-related class names, you are on the TE path.

If you switch from local to TE, keep in mind that you often need to train from scratch or convert checkpoints. TE is usually faster, but performance depends on hardware. Also, if modifying TE components is too complex for your changes, it can be practical to temporarily switch back to local for iteration and debugging.

### Where TE integration lives

Modifying TE operators in detail can be complex. A key file is:

- `megatron/core/extensions/transformer_engine.py`

This file is the core bridge between Megatron-LM and TE. It acts like a “translator” so Megatron can use TE’s fused kernels through Megatron-friendly interfaces. It wraps TE operators such as:

- `TENorm` (high-performance LayerNorm/RMSNorm)
- `TELinear` (basic Linear)
- `TEColumnParallelLinear` (column-parallel Linear)
- `TERowParallelLinear` (row-parallel Linear)
- `TELayerNormColumnParallelLinear` (fused LayerNorm + Linear)
- `TEDotProductAttention` (high-performance attention, e.g., FlashAttention-style)

It also includes comprehensive support for quantized training. For example:

- `TEQuantizationRecipe`: defines a single quantization “recipe” (FP8, FP4, or custom)
- `TEQuantizationParams`: allows different precision policies for training vs inference

In practice, you often replace components in a classic Transformer layer with these TE wrappers—for example, using `TEColumnParallelLinear` for the first MLP projection and `TERowParallelLinear` for the second. (The difference between Column and Row parallelism will be discussed later in the Megatron parallelism posts.)

More specifically, the first Linear is often replaced by `TELayerNormColumnParallelLinear`, because a LayerNorm is computed right before that Linear, and TE fuses them by default for better performance. One caveat: if you want to use your own custom TE Linear module, remember that you may need to support LayerNorm fusion as well—or decouple the fusion and explicitly add LayerNorm back.

---
## Summary

At this point, you should have a clearer picture of Megatron-LM’s complex architecture. Looking back at the confusion people feel when they first encounter this framework, many issues come from not understanding the design and the multiple parallel build paths. Megatron-LM is complex partly because it carries historical baggage: it does not aggressively clean up old systems. The new core system represents the future direction; the legacy system exists for backward compatibility.

Many beginners (including myself) have experienced: you follow a tutorial, modify some code, and the trained model doesn’t change at all. Later you realize you modified the core system, but your launch flags pointed to the legacy system. As Megatron grows, even more implementations get integrated, and “my modification doesn’t take effect” becomes a very common issue—so always verify which path you are using.

Megatron-LM’s modular design is actually elegant, but it requires layered thinking. For high-level changes (cross-layer connections, changing how attention and MLP are combined), focus on `TransformerBlock` / `TransformerLayer`. For low-level compute changes, modify the specific component modules.

Finally, writing code is not the same as succeeding—verification matters. Printing model structure quickly tells you whether changes took effect. Checking parameter count can reveal subtle issues (e.g., MLP structure changed but params did not). Running a small forward pass ensures the modified model at least executes. When issues arise, don’t immediately dive into thousands of lines of code—add a few well-placed print statements and trace parameter flow.

As an extra tip, exporting a structure graph via `torch.onnx.export()` can sometimes be more informative than pages of logs. Comparing checkpoint file sizes before and after modifications is another quick sanity check: if parameter count changes, checkpoint size should usually change as well.

---
## Closing thoughts

Beyond model construction, pretraining involves many other complex tasks: training hyperparameters, optimizer setup, learning-rate schedules, parallelism configuration, checkpointing and evaluation, and more. These processes may be complex, but they all build on the foundational concepts discussed here. Like building a house, if the foundation is unstable, no amount of fancy decoration will save it.

Megatron-LM is genuinely complex, and it is normal to feel overwhelmed at the beginning. But as emphasized throughout this post: understanding the architecture matters more than memorizing APIs, and verifying modifications matters more than trusting your intuition. If you keep these two principles in mind and iterate with patience, you can eventually find your way through this code maze.

I hope this post helps you if you are currently “wrestling” with Megatron-LM, and helps you avoid some of the detours we have taken. Remember: every pitfall is a learning opportunity—the key is to learn from it and avoid falling into the same hole twice.
