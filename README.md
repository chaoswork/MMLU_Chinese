# MMLU-Chinese(Measuring Massive Multitask Language Understanding 中文翻译版)

最近大模型能力的测评越来越重要，其中MMLU是一个很关键的数据集合。但是MMLU只能提供英文的测评，如果测试想测试中文的能力怎么办？我这里给出了一个很取巧的方案，就是将MMLU的问题和答案翻译成中文，这样就可以测试中文大模型的能力。

# ⚠️ 注意

- 本项目还比较粗糙，测评得分仅供参考。

## 目前的进度：

- 一共有57个类别。
- Fewshot的数量: 57 * 5。目前已经整理完成
- 测试问题数量: 57 * 10。目前每个类别的问题只完成了10个，每个类别的问题100-200个不等。所以目前只完成了其中一小部分。

# Models support matrix

## 目前支持如下的模型
| Model        | Support | 执行命令| score(English prompt) | score(中文Prompt)
|--------------| ---- |------------|------|--------|
| LLaMA        | ✅  |CUDA_VISIBLE_DEVICES=0 python evaluate_llama.py -m decapoda-research/llama-7b-hf -s llama_result |0.28|0.297|
| Bloomz        | ✅  | CUDA_VISIBLE_DEVICES=0 python evaluate_bloomz.py -m bigscience/bloomz-7b1-mt -s bloom_result |0.345|0.362|
| ChatGLM      | ✅  |CUDA_VISIBLE_DEVICES=0 python evaluate_chatglm.py -m THUDM/chatglm-6b -s chatglm_result |0.3|0.310|

- LLaMA模型目前还未放入transformer的主干，可以使用 [zphang的版本](https://github.com/zphang/transformers/tree/llama_push)

# 当前问题

- 由于是机翻+个人翻译，加上很多领域并不熟悉，所以可能会有错误。如果有错误欢迎指出。

# TODO

1. 翻译更多的问题
2. 多卡支持

# 交流

可关注下面公众号，回复"交流群"进群

[](./images/qrcode.jpg)

# 下面是Fork前的内容, 主要针对英文
This is the repository for [Measuring Massive Multitask Language Understanding](https://arxiv.org/pdf/2009.03300) by
[Dan Hendrycks](https://people.eecs.berkeley.edu/~hendrycks/), [Collin Burns](http://collinpburns.com), [Steven Basart](https://stevenbas.art), [Andy Zou](https://andyzoujm.github.io/), Mantas Mazeika, [Dawn Song](https://people.eecs.berkeley.edu/~dawnsong/), and [Jacob Steinhardt](https://www.stat.berkeley.edu/~jsteinhardt/) (ICLR 2021).

This repository contains OpenAI API evaluation code, and the test is available for download [**here**](https://people.eecs.berkeley.edu/~hendrycks/data.tar).

## Test Leaderboard

If you want to have your model added to the leaderboard, please reach out to us or submit a pull request.


Results of the test:
|                Model               | Authors |  Humanities |  Social Sciences  | STEM | Other | Average |
|------------------------------------|----------|:-------:|:-------:|:-------:|:-------:|:-------:|
| [Chinchilla](https://arxiv.org/abs/2203.15556) (70B, few-shot) | Hoffmann et al., 2022 | 63.6 | 79.3 | 54.9 | 73.9 | 67.5
| [Gopher](https://storage.googleapis.com/deepmind-media/research/language-research/Training%20Gopher.pdf) (280B, few-shot) | Rae et al., 2021 | 56.2 | 71.9 | 47.4 | 66.1 | 60.0
| [GPT-3](https://arxiv.org/abs/2005.14165) (175B, fine-tuned) | Brown et al., 2020 | 52.5 | 63.9 | 41.4 | 57.9 | 53.9
| [flan-T5-xl](https://arxiv.org/abs/2210.11416) | Chung et al., 2022 | 46.3 | 57.7 | 39.0 | 55.1 | 49.3
| [UnifiedQA](https://arxiv.org/abs/2005.00700) | Khashabi et al., 2020 | 45.6 | 56.6 | 40.2 | 54.6 | 48.9
| [GPT-3](https://arxiv.org/abs/2005.14165) (175B, few-shot) | Brown et al., 2020 | 40.8 | 50.4 | 36.7 | 48.8 | 43.9
| [GPT-3](https://arxiv.org/abs/2005.14165) (6.7B, fine-tuned) | Brown et al., 2020 | 42.1 | 49.2 | 35.1 | 46.9 | 43.2
| [flan-T5-large](https://arxiv.org/abs/2210.11416) | Chung et al., 2022 | 39.1 | 49.1 | 33.2 | 47.4 | 41.9
| [flan-T5-base](https://arxiv.org/abs/2210.11416) | Chung et al., 2022 | 34.0 | 38.1 | 27.6 | 37.0 | 34.2
| [GPT-2](https://arxiv.org/abs/2005.14165) | Radford et al., 2019 | 32.8 | 33.3 | 30.2 | 33.1 | 32.4
| [flan-T5-small](https://arxiv.org/abs/2210.11416) | Chung et al., 2022 | 29.9 | 30.9 | 27.5 | 29.7 | 29.5
| Random Baseline           | N/A | 25.0 | 25.0 | 25.0 | 25.0 | 25.0 | 25.0


## Citation

Please cite us when using our code, data or model.

```
@misc{MMLU_Chinese,
  author = {Huang Chao},
  title = {Measuring Massive Multitask Language Understanding in Chinese},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/chaoswork/MMLU_Chinese}},
}
```

If you find this useful in your research, please consider citing the test and also the [ETHICS](https://arxiv.org/abs/2008.02275) dataset it draws from:

    @article{hendryckstest2021,
      title={Measuring Massive Multitask Language Understanding},
      author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
      journal={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2021}
    }

    @article{hendrycks2021ethics,
      title={Aligning AI With Shared Human Values},
      author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
      journal={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2021}
    }
