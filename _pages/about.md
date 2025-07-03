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

Hi, I'm Ruizhe Wang (王瑞哲), currently pursuing a joint Ph.D. program between [University of Science and Technology and China (USTC)](https://en.ustc.edu.cn/) and [Microsoft Research Asia (MSRA)](https://www.microsoft.com/en-us/research/lab/microsoft-research-asia/), co-supervised by Professor [Zhengjun Zha (查正军)](http://en.auto.ustc.edu.cn/2021/0616/c26828a513174/page.htm) and Professor [Baining Guo (郭百宁)](https://www.microsoft.com/en-us/research/people/bainguo/). I received my Bachelor of Engineering degree in Electronic Information Engineering from the University of Science and Technology of China. I also collaborate with [Peng Cheng (程鹏)](https://cp5555.github.io/) and [Yeyun Gong (宫叶云)](https://www.microsoft.com/en-us/research/people/yegong/) from Microsoft Research Asia.

<!-- <img src='./images/microsoft_logo.svg' style="width: 4em;"> -->

My research interest are AI Infrastructure, Large Language Model (LLM) Pretraining, and Efficient AI System Designing. I enjoy building scalable models, exploring novel training methods, and solving challenging problems at the intersection of theory and practice. Currently, I work on developing high-performance machine learning infrastructure and advancing the capabilities of large language models.

In my free time, I enjoy sharing knowledge, contributing to open-source projects, and staying curious about emerging technologies. For more info related to my research, please refer to my <a href='https://scholar.google.com/citations?user=gu_oKFIAAAAJ'>google scholar homepage</a> <a href='https://scholar.google.com/citations?user=gu_oKFIAAAAJ'><img src="https://img.shields.io/endpoint?url={{ url | url_encode }}&logo=Google%20Scholar&labelColor=f6f6f6&color=9cf&style=flat&label=citations"></a>.


# 📖 Educations
- *2023.09 - now*, PhD Candidate, University of Science and Technology of China, Department of Automation 
- *2019.09 - 2023.06*, Bachelor's Degree, University of Science and Technology of China, Department of Electronic Engineering and Information Science
- *2016.09 - 2019.06*, Hefei No.fifty Middle School

<!-- 
# 🔥 News
- *2022.02*: &nbsp;🎉🎉 Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. 
- *2022.02*: &nbsp;🎉🎉 Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet.  -->

# 🦈 Blogs
- (in Chinese) 🔥 [10,000-word analysis of FP4 training of large language models: Optimizing Large Language Model Training Using FP4 Quantization Paper Sharing](https://zhuanlan.zhihu.com/p/1910768186401466238)


# 📝 Publications 

<!-- please make sure all paper images has 1200px width -->

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICML 2025</div><img src='images/fp4_frame_thin.png' alt="sym" width="100%"></div></div>
<!-- <div class='paper-box'><div class='paper-box-image'><div><img src='images/fp4_frame_thin.png' alt="sym" width="450px" style="object-fit: contain;"></div></div> -->
<div class='paper-box-text' markdown="1">

[Optimizing Large Language Model Training Using FP4 Quantization](https://arxiv.org/abs/2501.17116)

**Ruizhe Wang**, Yeyun Gong, Xiao Liu, Guoshuai Zhao, Ziyue Yang, Baining Guo, Zhengjun Zha, Peng Cheng

<a href="https://github.com/Azure/MS-AMP"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"></a> 
<!-- [**Code**](https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=DhtAFkwAAAAJ&citation_for_view=DhtAFkwAAAAJ:ALROH1vI_8AC) <strong><span class='show_paper_citations' data='DhtAFkwAAAAJ:ALROH1vI_8AC'></span></strong> -->
<!-- - Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet.  -->
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><img src='images/fp8_frame_thin.png' alt="sym" width="100%"></div></div>
<!-- <div class='paper-box'><div class='paper-box-image'><div><img src='images/fp8_frame_thin.png' alt="sym" width="450px" style="object-fit: contain;"></div></div> -->
<div class='paper-box-text' markdown="1">

[FP8-LM: Training FP8 Large Language Models](https://arxiv.org/abs/2310.18313)

Houwen Peng, Kan Wu, Yixuan Wei, Guoshuai Zhao, Yuxiang Yang, Ze Liu, Yifan Xiong, Ziyue Yang, Bolin Ni, Jingcheng Hu, Ruihang Li, Miaosen Zhang, Chen Li, Jia Ning, **Ruizhe Wang**, Zheng Zhang, Shuguang Liu, Joe Chau, Han Hu, Peng Cheng

<a href="https://github.com/Azure/MS-AMP"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"></a> 
</div>
</div>

# 🏆 Honors and Awards
- *2023.06* : Outstanding graduates of USTC.
- *2022.09* : China National Scholarship. 
- *2019.09/2020.09/2021.09* : Scholarships for the Elites Program of USTC.  

<!-- # 💬 Invited Talks
- *2021.06*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. 
- *2021.03*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet.  \| [\[video\]](https://github.com/) -->

# 💻 Internships
- *2022.07 - now*, Research Intern in [Microsoft Research Asia (MSRA)](https://www.microsoft.com/en-us/research/lab/microsoft-research-asia/), Beijing, China.
