---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

<div class="about-intro">
<p>
Hi, I'm <span class="intro-highlight">Ruizhe Wang (王瑞哲)</span> 👋, currently pursuing a joint Ph.D. program between <a href="https://en.ustc.edu.cn/">USTC</a> and <a href="https://www.microsoft.com/en-us/research/lab/microsoft-research-asia/">Microsoft Research Asia</a>, co-supervised by Prof. <a href="http://en.auto.ustc.edu.cn/2021/0616/c26828a513174/page.htm">Zhengjun Zha</a> and Prof. <a href="https://www.microsoft.com/en-us/research/people/bainguo/">Baining Guo</a>. I also collaborate closely with <a href="https://cp5555.github.io/">Peng Cheng</a> and <a href="https://www.microsoft.com/en-us/research/people/yegong/">Yeyun Gong</a> at MSRA.
</p>

<p>
My research interest are AI Infrastructure, Large Language Model (LLM) Pretraining, and Efficient AI System Designing. I enjoy building scalable models, exploring novel training methods, and solving challenging problems at the intersection of theory and practice.
</p>

<div class="research-interests">
<span class="interest-tag"><i class="fa fa-server"></i> AI Infrastructure</span>
<span class="interest-tag"><i class="fa fa-brain"></i> LLM Pretraining</span>
<span class="interest-tag"><i class="fa fa-bolt"></i> Efficient AI Systems</span>
<span class="interest-tag"><i class="fa fa-microchip"></i> Low-bit Quantization</span>
</div>

<p>
I’m looking to collaborate on inspirable companions. You can find more info related to my research on my <a href='https://scholar.google.com/citations?user=gu_oKFIAAAAJ'>google scholar homepage</a>: <a href='https://scholar.google.com/citations?user=gu_oKFIAAAAJ'><img src="https://img.shields.io/endpoint?logo=Google%20Scholar&url=https%3A%2F%2Fcdn.jsdelivr.net%2Fgh%2FMr-Philo%2Fmr-philo.github.io@google-scholar-stats%2Fgs_data_shieldsio.json&labelColor=f6f6f6&color=9cf&style=flat&label=citations"></a>
</p>

</div>

# 📖 Educations

<div class="timeline">

<div class="timeline-item">
<span class="timeline-date">2023.09 - Present</span>
<div class="timeline-title">Ph.D. Candidate in Automation</div>
<div class="timeline-subtitle">University of Science and Technology of China (USTC)</div>
<div class="timeline-location"><i class="fa fa-map-marker-alt"></i> Hefei, China · Joint Program with Microsoft Research Asia</div>
</div>

<div class="timeline-item">
<span class="timeline-date">2019.09 - 2023.06</span>
<div class="timeline-title">B.E. in Electronic Information Engineering</div>
<div class="timeline-subtitle">University of Science and Technology of China (USTC)</div>
<div class="timeline-location"><i class="fa fa-map-marker-alt"></i> Hefei, China</div>
</div>

</div>


# 📝 Publications 

<div class='pub-box'>
<div class='pub-box-image'>
<div class="pub-venue">ICML 2025</div>
<a href="https://arxiv.org/abs/2501.17116"><img src='images/publications/fp4_frame_thin.png' alt="FP4 Quantization"></a>
</div>
<div class='pub-box-text' markdown="1">

### [Optimizing Large Language Model Training Using FP4 Quantization](https://arxiv.org/abs/2501.17116)

<p class="pub-authors"><strong>Ruizhe Wang</strong>, Yeyun Gong, Xiao Liu, Guoshuai Zhao, Ziyue Yang, Baining Guo, Zhengjun Zha, Peng Cheng</p>

<div class="pub-meta">
<span><i class="fa fa-calendar"></i> Jan 2025</span>
<span><i class="fa fa-book"></i> ICML 2025</span>
</div>

<p class="pub-excerpt">We propose the first FP4 training framework for LLMs, introducing Differentiable Gradient Estimation and Outlier Clamp and Compensation to address quantization challenges, achieving lossless pre-training performance on 13B LLMs and 100B tokens datasets.</p>

<div class="pub-links">
<a href="https://arxiv.org/abs/2501.17116"><img src="https://img.shields.io/badge/arXiv-2501.17116-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white"></a>
<a href="https://github.com/Azure/MS-AMP"><img src="https://img.shields.io/badge/GitHub-Code-100000?style=for-the-badge&logo=github&logoColor=white"></a>
</div>

</div>
</div>

<div class='pub-box'>
<div class='pub-box-image'>
<a href="https://arxiv.org/abs/2510.08008"><img src='images/publications/recycle-frame.png' alt="Recycling Checkpoints"></a>
</div>
<div class='pub-box-text' markdown="1">

### [Recycling Pretrained Checkpoints: Orthogonal Growth of MoE for Efficient LLM Pre-Training](https://arxiv.org/abs/2510.08008)

<p class="pub-authors"><strong>Ruizhe Wang</strong>, Yucheng Ding, Xiao Liu, Yaoxiang Wang, Peng Cheng, Baining Guo, Zhengjun Zha, Yeyun Gong</p>

<div class="pub-meta">
<span><i class="fa fa-calendar"></i> Oct 2025</span>
<span><i class="fa fa-file-alt"></i> Preprint</span>
</div>

<p class="pub-excerpt">We propose a "checkpoint recycling" strategy that expands existing models through orthogonal growth on 70B MoE models with 1T training tokens, delivering a 10.6% accuracy improvement over training from scratch while significantly maximizing the value of prior computational investments.</p>

<div class="pub-links">
<a href="https://arxiv.org/abs/2510.08008"><img src="https://img.shields.io/badge/arXiv-2510.08008-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white"></a>
</div>

</div>
</div>

<div class='pub-box'>
<div class='pub-box-image'>
<a href="https://arxiv.org/abs/2509.26520"><img src='images/publications/Matryoshka-frame.png' alt="Matryoshka MoE"></a>
</div>
<div class='pub-box-text' markdown="1">

### [Training Matryoshka Mixture-of-Experts for Elastic Inference-Time Expert Utilization](https://arxiv.org/abs/2509.26520)

<p class="pub-authors">Yaoxiang Wang, Qingguo Hu, Yucheng Ding, <strong>Ruizhe Wang</strong>, Yeyun Gong, Jian Jiao, Yelong Shen, Peng Cheng, Jinsong Su</p>

<div class="pub-meta">
<span><i class="fa fa-calendar"></i> Sep 2025</span>
<span><i class="fa fa-file-alt"></i> Preprint</span>
</div>

<p class="pub-excerpt">Introducing Matryoshka MoE (M-MoE) that enables elastic expert utilization during inference by instilling a coarse-to-fine structure into expert ensembles, allowing dynamic compute allocation based on resource constraints while maintaining model quality.</p>

<div class="pub-links">
<a href="https://arxiv.org/abs/2509.26520"><img src="https://img.shields.io/badge/arXiv-2509.26520-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white"></a>
</div>

</div>
</div>


# 📝 Technical Reports 

<div class='pub-box'>
<div class='pub-box-image'>
<a href="https://arxiv.org/abs/2310.18313"><img src='images/publications/fp8_frame_thin.png' alt="FP8-LM"></a>
</div>
<div class='pub-box-text' markdown="1">

### [FP8-LM: Training FP8 Large Language Models](https://arxiv.org/abs/2310.18313)

<p class="pub-authors">Houwen Peng*, Kan Wu*, Yixuan Wei*, Guoshuai Zhao, Yuxiang Yang, Ze Liu, Yifan Xiong, Ziyue Yang, Bolin Ni, Jingcheng Hu, Ruihang Li, Miaosen Zhang, Chen Li, Jia Ning, <strong>Ruizhe Wang</strong>, Zheng Zhang, Shuguang Liu, Joe Chau, Han Hu, Peng Cheng</p>

<div class="pub-meta">
<span><i class="fa fa-calendar"></i> Oct 2023</span>
<span><i class="fa fa-file-alt"></i> Technical Report</span>
</div>

<p class="pub-excerpt">A comprehensive FP8 automatic mixed-precision training framework for LLMs that achieves up to 39% reduction in memory usage and 1.75× training speedup on H100 GPUs while maintaining model accuracy comparable to BF16 training.</p>

<div class="pub-links">
<a href="https://arxiv.org/abs/2310.18313"><img src="https://img.shields.io/badge/arXiv-2310.18313-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white"></a>
<a href="https://github.com/Azure/MS-AMP"><img src="https://img.shields.io/badge/GitHub-Code-100000?style=for-the-badge&logo=github&logoColor=white"></a>
</div>

</div>
</div>

<div class='pub-box'>
<div class='pub-box-image'>
<a href="https://arxiv.org/abs/2512.13488"><img src='images/publications/sigma-moe-frame.png' alt="SIGMA"></a>
</div>
<div class='pub-box-text' markdown="1">

### [SIGMA: An AI-Empowered Training Stack on Early-Life Hardware](https://arxiv.org/abs/2512.13488)

<p class="pub-authors">Lei Qu, Lianhai Ren, Peng Cheng, Rui Gao, <strong>Ruizhe Wang</strong>, Tianyu Chen, Xiao Liu, Xingjian Zhang, Yeyun Gong, Yifan Xiong, Yucheng Ding, Yuting Jiang, Zhenghao Lin, Zhongxin Guo, Ziyue Yang</p>

<div class="pub-meta">
<span><i class="fa fa-calendar"></i> Dec 2025</span>
<span><i class="fa fa-file-alt"></i> Technical Report</span>
</div>

<p class="pub-excerpt">Introducing SIGMA, an open-source training stack designed to overcome the reliability and efficiency challenges of large-scale AI training on early-life accelerators, enabling the stable pre-training of a 200B MoE model with 94.45% accelerator utilization. </p>

<div class="pub-links">
<a href="https://arxiv.org/abs/2512.13488"><img src="https://img.shields.io/badge/arXiv-2512.13488-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white"></a>
<a href="https://github.com/microsoft/LuciaTrainingPlatform"><img src="https://img.shields.io/badge/GitHub-Code-100000?style=for-the-badge&logo=github&logoColor=white"></a>
</div>

</div>
</div>

<div class='pub-box'>
<div class='pub-box-image'>
<a href="https://qghuxmu.github.io/Sigma-MoE-Tiny/"><img src='images/publications/sigma-moe-tiny-frame.png' alt="Sigma-MoE-Tiny"></a>
</div>
<div class='pub-box-text' markdown="1">

### [Sigma-MoE-Tiny Technical Report](https://arxiv.org/abs/2512.16248)

<p class="pub-authors">Qingguo Hu, Zhenghao Lin, Ziyue Yang, Yucheng Ding, Xiao Liu, Yuting Jiang, <strong>Ruizhe Wang</strong>, Tianyu Chen, Zhongxin Guo, Yifan Xiong, Rui Gao, Lei Qu, Jinsong Su, Peng Cheng, Yeyun Gong</p>

<div class="pub-meta">
<span><i class="fa fa-calendar"></i> Dec 2025</span>
<span><i class="fa fa-file-alt"></i> Technical Report</span>
</div>

<p class="pub-excerpt">Introducing Sigma-MoE-Tiny, an ultra-sparse MoE language model that activates only 0.5B out of 20B parameters per token by using a fine-grained 96-expert architecture, achieving state-of-the-art performance at this extreme sparsity.</p>

<div class="pub-links">
<a href="https://arxiv.org/abs/2512.16248"><img src="https://img.shields.io/badge/arXiv-2512.16248-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white"></a>
<a href="https://github.com/microsoft/ltp-megatron-lm"><img src="https://img.shields.io/badge/GitHub-Code-100000?style=for-the-badge&logo=github&logoColor=white"></a>
<a href="https://qghuxmu.github.io/Sigma-MoE-Tiny/"><img src="https://img.shields.io/badge/Homepage-blue?style=for-the-badge&logo=googlechrome&logoColor=white"></a>
</div>

</div>
</div>


# 🦈 Blogs

> 📚 View full blogs page: [All Blog Posts](/blog-posts/)

<div class='blog-box'>
<div class='blog-box-image'><a href="/posts/2025/05/analysis-fp4/"><img src='images/publications/fp4_frame_thin.png' alt="FP4 Quantization"></a></div>
<div class='blog-box-text' markdown="1">

### [5,000 words Analysis of FP4 Quantization for Training LLMs](/posts/2025/05/analysis-fp4/)

<div class="blog-meta">
<span><i class="fa fa-calendar"></i> May 30, 2025</span>
<span><i class="fa fa-clock"></i> 29 min read</span>
</div>

<p class="blog-excerpt">Detailed Paper Interpretation of "Optimizing Large Language Model Training Using FP4 Quantization". This post walks you through the motivation, key insights, and design rationale behind our work.</p>

<div class="blog-tags"><span class="tag">Quantization</span><span class="tag">Paper Interpretation</span></div>

</div>
</div>

<div class='blog-box'>
<div class='blog-box-image'><a href="/posts/2025/08/quantization-scaling-law/"><img src='images/posts/2025-08-03-quantization-scaling-law/quantization-scaling-law.png' alt="Quantization Scaling Law"></a></div>
<div class='blog-box-text' markdown="1">

### [A One-Stop Guide to Scaling Laws in LLM Quantization](/posts/2025/08/quantization-scaling-law/)

<div class="blog-meta">
<span><i class="fa fa-calendar"></i> Aug 3, 2025</span>
<span><i class="fa fa-clock"></i> 27 min read</span>
</div>

<p class="blog-excerpt">A comprehensive overview of Quantization Scaling Laws. Dive deep into 5 papers to understand how performance loss from quantization varies with model parameters and token count.</p>

<div class="blog-tags"><span class="tag">Quantization</span><span class="tag">Scaling Laws</span></div>

</div>
</div>

<div class='blog-box'>
<div class='blog-box-image'><a href="/posts/2025/10/megatron-exp-0/"><img src='images/posts/2025-10-10-megatron-exp-0/parallelism-deepspeed-3d.png' alt="Megatron-LM Guide"></a></div>
<div class='blog-box-text' markdown="1">

### [Megatron-LM Training Large Models Practical Guide: 0 - Preface](/posts/2025/10/megatron-exp-0/)

<div class="blog-meta">
<span><i class="fa fa-calendar"></i> Oct 10, 2025</span>
<span><i class="fa fa-clock"></i> 16 min read</span>
</div>

<p class="blog-excerpt">Why we must use Megatron-LM for large model training, and some warnings for those who have never used it before. A practical guide from personal experience.</p>

<div class="blog-tags"><span class="tag">Megatron-LM</span><span class="tag">Practical Guide</span></div>

</div>
</div>

<div class='blog-box'>
<div class='blog-box-image'><a href="/posts/2025/10/looped-1"><img src='images/posts/2025-10-28-recursive-paper-reading/recursive-cover.png' alt="Recursive Transformers"></a></div>
<div class='blog-box-text' markdown="1">

### [Paper Summary for Recursive Looped Transformers: Parameter Efficiency](/posts/2025/10/looped-1)

<div class="blog-meta">
<span><i class="fa fa-calendar"></i> Oct 28, 2025</span>
<span><i class="fa fa-clock"></i> 25 min read</span>
</div>

<p class="blog-excerpt">Exploring how loops and recursion can improve parameter utilization efficiency in LLMs. A comprehensive summary of recursive mechanisms in Transformer architectures.</p>

<div class="blog-tags"><span class="tag">Recursive Transformers</span><span class="tag">Paper Interpretation</span></div>

</div>
</div>


# 🏆 Honors and Awards

<div class="awards-grid">

<div class="award-card">
<div class="award-icon">🎓</div>
<span class="award-year">2023</span>
<div class="award-title">Outstanding Graduate of USTC</div>
</div>

<div class="award-card">
<div class="award-icon">🏅</div>
<span class="award-year">2022</span>
<div class="award-title">China National Scholarship</div>
</div>

<div class="award-card">
<div class="award-icon">⭐</div>
<span class="award-year">2019-2021</span>
<div class="award-title">Elites Program Scholarship (3×)</div>
</div>

</div>


# 💻 Internships

<div class="internship-card">
<div class="internship-logo">
<i class="fab fa-microsoft"></i>
</div>
<div class="internship-content">
<div class="internship-company"><a href="https://www.microsoft.com/en-us/research/lab/microsoft-research-asia/">Microsoft Research Asia (MSRA)</a></div>
<div class="internship-role">Research Intern · Natural Language Computing Group</div>
<div class="internship-meta">
<span><i class="fa fa-calendar"></i> Jul 2022 - Present</span>
<span><i class="fa fa-map-marker-alt"></i> Beijing, China</span>
</div>
</div>
</div>
