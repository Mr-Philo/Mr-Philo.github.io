---
title: 'Paper Reading Session for Recursive Looped Transformers: Parameter Efficiency'
date: 2025-10-28
permalink: /posts/2025/10/looped-1
tags:
  - Recursive Transformers
  - Paper Interpretation
---

论文串读：大模型中的循环递归（Recursive Looped Transformers）之参数高效性

循环（Loop）和递归（Recursion）一直是深度学习中的重要技巧，比如我们所熟知的循环神经网络RNN（Recursive Neural Networks），相较于传统的神经网络，其优势在于可以通过隐状态在时间维度上传递信息，使模型能够捕捉上下文中的时序依赖关系。而在大语言模型时代，研究者们又进一步探索了循环和递归机制在如今越来越大的模型中能起到什么作用。虽然Scaling Law缩放定律告诉我们，大语言模型的最后表现与其参数量多少和所训练的语料数量息息相关。但与此同时，递归的策略也能在LLM的场景中体现重要的作用。

最近一些关于在LLM中应用递归机制的文章，其可以大致分为两个主要板块：一是提高参数利用率，使得单个参数更加高效。通过将特定参数循环计算多次，使得模型实际运算量达到一个比较高的水平，但实际上原有模型的真实参数量并没有改变，因此可以做到在不增加参数量的情况下将计算量scale up上去。二是提升模型的最终表现，尤其是推理能力。对某些层进行多次递归计算，可以在隐空间内模拟长上下文思维链（Long-context Chain of Thoughts）的场景，以最终提升模型在下游推理任务上的表现。这两个大板块各有侧重，但也相辅相成，通过递归计算既可以让原有模型在参数量不变的情况下，提升最后下游任务的准确率，也可以让模型在最终任务表现不变的情况下，降低其所需要的参数数量。

为此这篇文章先介绍几篇关于参数高效性的文章，之后会再单开一篇来介绍关于提升模型推理能力的文章。本篇文章还会简要介绍递归机制在Transformer架构中的引入。

Paper list:

2018.07.10 (last updated: 2019.03.05) Universal Transformers (ICLR 2019)  https://arxiv.org/abs/1807.03819

2021.04.13 (last updated: 2023.06.02) Lessons on Parameter Sharing across Layers in Transformers (SustaiNLP 2023) https://arxiv.org/abs/2104.06022

2024.10.28 (last updated: 2025.02.28) Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA (ICLR 2025) https://arxiv.org/abs/2410.20672

2025.07.14 (last updated: 2025.07.21) Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation (ICML 2025) https://arxiv.org/abs/2507.10524

2025.06.26 (last updated: 2025.08.04) Hierarchical Reasoning Model https://arxiv.org/abs/2506.21734

2025.10.06 Less is More: Recursive Reasoning with Tiny Networks https://arxiv.org/abs/2510.04871


