# 所有RAG技术：更简单、实用的入门方法 ✨

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Nebius AI](https://img.shields.io/badge/Nebius%20AI-API-brightgreen)](https://cloud.nebius.ai/services/llm-embedding) [![OpenAI](https://img.shields.io/badge/OpenAI-API-lightgrey)](https://openai.com/) [![Medium](https://img.shields.io/badge/Medium-Blog-black?logo=medium)](https://medium.com/@fareedkhandev/testing-every-rag-technique-to-find-the-best-094d166af27f)

本仓库以清晰、实用的方式介绍**检索增强生成（RAG）**，将高级技术拆解为简单易懂的实现。不依赖如 `LangChain` 或 `FAISS` 等框架，所有内容均基于常用的 Python 库（如 `openai`、`numpy`、`matplotlib` 等）构建。

目标很简单：提供可读性强、易于修改、具有教育意义的代码。通过聚焦基础原理，本项目帮助大家深入理解RAG的实际工作方式。

## 更新: 📢
- （2025年5月12日）新增了一个关于如何使用知识图谱处理大数据的Notebook。
- （2025年4月27日）新增了一个Notebook，用于针对给定查询寻找最佳RAG技术（简单RAG + 重排序器 + 查询重写）。
- （2025年3月20日）新增了一个关于强化学习RAG的Notebook。
- （2025年3月7日）仓库新增20种RAG技术。

## 🚀 内容简介

本仓库包含一系列Jupyter Notebook，每个Notebook聚焦于一种特定的RAG技术。每个Notebook都提供：

*   技术的简明解释。
*   从零开始的分步实现。
*   带有行内注释的清晰代码示例。
*   评估与对比，展示技术效果。
*   可视化结果。

以下是所涵盖技术的简要一览：

| Notebook                                      | 描述                                                                                                                                                         |
| :-------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [1. Simple RAG](1_simple_rag.ipynb)           | 基础RAG实现，入门首选！                                                                                                                               |
| [2. Semantic Chunking](2_semantic_chunking.ipynb) | 基于语义相似性切分文本，获得更有意义的片段。                                                                                                         |
| [3. Chunk Size Selector](3_chunk_size_selector.ipynb) | 探索不同切分大小对检索性能的影响。                                                                                                                  |
| [4. Context Enriched RAG](4_context_enriched_rag.ipynb) | 检索相邻片段以提供更多上下文。                                                                                                                     |
| [5. Contextual Chunk Headers](5_contextual_chunk_headers_rag.ipynb) | 在嵌入前为每个片段添加描述性标题。                                                                                                              |
| [6. Document Augmentation RAG](6_doc_augmentation_rag.ipynb) | 从文本片段生成问题以增强检索过程。                                                                                                               |
| [7. Query Transform](7_query_transform.ipynb)   | 重写、扩展或分解查询以提升检索效果。包含**Step-back Prompting**和**子查询分解**。                                      |
| [8. Reranker](8_reranker.ipynb)               | 使用LLM对初步检索结果进行重排序，提高相关性。                                                                                                   |
| [9. RSE](9_rse.ipynb)                         | 相关片段提取：识别并重构连续文本片段，保留上下文。                                                                                             |
| [10. Contextual Compression](10_contextual_compression.ipynb) | 实现上下文压缩，过滤并压缩检索片段，最大化相关信息。                                                                                       |
| [11. Feedback Loop RAG](11_feedback_loop_rag.ipynb) | 融入用户反馈，持续学习和改进RAG系统。                                                                                                      |
| [12. Adaptive RAG](12_adaptive_rag.ipynb)     | 根据查询类型动态选择最佳检索策略。                                                                                                            |
| [13. Self RAG](13_self_rag.ipynb)             | 实现Self-RAG，动态决定何时及如何检索，评估相关性、支持性和实用性。                                                                 |
| [14. Proposition Chunking](14_proposition_chunking.ipynb) | 将文档拆分为原子、事实性陈述，实现精确检索。                                                                                           |
| [15. Multimodel RAG](15_multimodel_rag.ipynb)   | 文本与图片联合检索，使用LLaVA为图片生成描述。                                                                                   |
| [16. Fusion RAG](16_fusion_rag.ipynb)         | 向量检索与关键词（BM25）检索融合，提升效果。                                                                                              |
| [17. Graph RAG](17_graph_rag.ipynb)           | 以图结构组织知识，实现相关概念的遍历。                                                                                                   |
| [18. Hierarchy RAG](18_hierarchy_rag.ipynb)        | 构建分层索引（摘要+详细片段），高效检索。                                                                                              |
| [19. HyDE RAG](19_HyDE_rag.ipynb)             | 利用假设文档嵌入提升语义匹配。                                                                                                         |
| [20. CRAG](20_crag.ipynb)                     | 校正型RAG：动态评估检索质量，必要时使用网络搜索兜底。                                                                              |
| [21. Rag with RL](21_rag_with_rl.ipynb)                     | 通过强化学习最大化RAG模型的奖励。                                                                                      |
| [Best RAG Finder](best_rag_finder.ipynb)     | 利用简单RAG+重排序器+查询重写，为给定查询寻找最佳RAG技术。                                                                |
| [22. Big Data with Knowledge Graphs](22_Big_data_with_KG.ipynb) | 使用知识图谱处理大规模数据集。                                                                                         |

## 🗂️ 仓库结构

```
fareedkhan-dev-all-rag-techniques/
├── README.md                          <- 你正在阅读的文件！
├── 1_simple_rag.ipynb
├── 2_semantic_chunking.ipynb
├── 3_chunk_size_selector.ipynb
├── 4_context_enriched_rag.ipynb
├── 5_contextual_chunk_headers_rag.ipynb
├── 6_doc_augmentation_rag.ipynb
├── 7_query_transform.ipynb
├── 8_reranker.ipynb
├── 9_rse.ipynb
├── 10_contextual_compression.ipynb
├── 11_feedback_loop_rag.ipynb
├── 12_adaptive_rag.ipynb
├── 13_self_rag.ipynb
├── 14_proposition_chunking.ipynb
├── 15_multimodel_rag.ipynb
├── 16_fusion_rag.ipynb
├── 17_graph_rag.ipynb
├── 18_hierarchy_rag.ipynb
├── 19_HyDE_rag.ipynb
├── 20_crag.ipynb
├── 21_rag_with_rl.ipynb
├── 22_big_data_with_KG.ipynb
├── best_rag_finder.ipynb
├── requirements.txt                   <- Python依赖
└── data/
    └── val.json                       <- 验证用示例数据（查询与答案）
    └── AI_Information.pdf             <- 测试用示例PDF文档
    └── attention_is_all_you_need.pdf  <- 测试多模态RAG的示例PDF
```

## 🛠️ 快速开始

1.  **克隆仓库：**

    ```bash
    git clone https://github.com/FareedKhan-dev/all-rag-techniques.git
    cd all-rag-techniques
    ```

2.  **安装依赖：**

    ```bash
    pip install -r requirements.txt
    ```

3.  **设置OpenAI API密钥：**

    *   从 [Nebius AI](https://studio.nebius.com/) 获取API密钥。
    *   将API密钥设置为环境变量：
        ```bash
        export OPENAI_API_KEY='YOUR_NEBIUS_AI_API_KEY'
        ```
        或
        ```bash
        setx OPENAI_API_KEY "YOUR_NEBIUS_AI_API_KEY"  # Windows下
        ```
        或在Python脚本/Notebook中：

        ```python
        import os
        os.environ["OPENAI_API_KEY"] = "YOUR_NEBIUS_AI_API_KEY"
        ```

4.  **运行Notebook：**

    使用Jupyter Notebook或JupyterLab打开任意`.ipynb`文件。每个Notebook都是自包含的，可独立运行。建议按文件内顺序依次执行。

    **注意：** `data/AI_Information.pdf`为测试用示例文档，可替换为你自己的PDF。`data/val.json`包含用于评估的示例查询和理想答案。`attention_is_all_you_need.pdf`用于多模态RAG Notebook测试。

## 💡 核心概念

*   **嵌入（Embeddings）：** 文本的数值表示，捕捉语义信息。我们使用Nebius AI的嵌入API，部分Notebook也用`BAAI/bge-en-icl`模型。
*   **向量存储（Vector Store）：** 用于存储和检索嵌入的简单数据库。我们用NumPy自建`SimpleVectorStore`类，实现高效相似度计算。
*   **余弦相似度（Cosine Similarity）：** 衡量两个向量相似度的指标，值越高表示越相似。
*   **切分（Chunking）：** 将文本分割为更小、更易管理的片段。我们探索了多种切分策略。
*   **检索（Retrieval）：** 针对查询找到最相关的文本片段。
*   **生成（Generation）：** 利用大语言模型（LLM）基于检索到的上下文和用户查询生成回复。我们通过Nebius AI API使用`meta-llama/Llama-3.2-3B-Instruct`模型。
*   **评估（Evaluation）：** 评估RAG系统回复质量，通常与参考答案对比，或用LLM打分相关性。

## 🤝 贡献

欢迎贡献！