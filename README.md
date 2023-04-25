# WikiWhy
[Paper](https://openreview.net/pdf?id=vaxnu-Utr4l) | [Poster](https://github.com/matt-seb-ho/WikiWhy/blob/add_load_instructions/wikiwhy_iclr23_poster.pdf)

WikiWhy is a new benchmark for evaluating models' ability to **explain** *between* cause and effect. 
WikiWhy is a QA dataset built around the novel auxiliary task of explaining the answer to a "why" questions in natural language. 
It contains over 9,000 “why” question-answer-rationale triples, grounded on Wikipedia facts across a diverse set of topics. Each rationale is a set of supporting statements connecting the question to the answer. 

This paper was accepted as a top 5% paper with oral presentation to the International Conference on Learning Representations (ICLR 2023) in Kigali, Rwanda.

![Figure 1.](https://github.com/matt-seb-ho/WikiWhy/blob/add_load_instructions/figures/poster_figure.png)

## Dataset Usage
In light of data contamination concerns, and to prevent WikiWhy from inadvertently being included in pre-training corpora, we've separated WikiWhy's columns into separate files. 
We hope that separating the inputs and output labels can help preserve WikiWhy's value as a benchmark.
To load WikiWhy into a single dataframe, we provide a simple script in `/code/load_wikiwhy.py`. 
Either copy the function source or import into your code.

```python
from load_wikiwhy import load_wikiwhy
wikiwhy = load_wikiwhy(directory_path="../dataset/v1.1/")
```

## Updates
04/24/2023 Added paper and poster links; added code and instructions to easily load in WikiWhy
02/28/2023 Added dataset version 1.1
02/24/2023 Added dataset version 1.0

## Citation
```
@inproceedings{
    ho2023wikiwhy,
    title={WikiWhy: Answering and Explaining Cause-and-Effect Questions},
    author={Matthew Ho and Aditya Sharma and Justin Chang and Michael Saxon and Sharon Levy and Yujie Lu and William Yang Wang},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=vaxnu-Utr4l}
}
```
