# XAI
Genetic programming method for explaining complex ensembles

## Overview
Interpreting state-of-the-art machine learning algorithms can be difficult. For example, why does a complex ensemble predict a particular class? Existing approaches to interpretable machine learning tend to be either local in their explanations, apply only to a particular algorithm, or overly complex in their global explanations. In this work, we propose a global model extraction method which uses multi-objective genetic programming to construct accurate, simplistic and model-agnostic representations of complex black-box estimators. We found the resulting representations are far simpler than existing approaches while providing comparable reconstructive performance. This is demonstrated on a range of datasets, by approximating the knowledge of complex black-box models such as 200 layer neural networks and ensembles of 500 trees, with a single tree.


## How to use

The model is a sklearn classifier, as such uses the sklearn api

```python
from src.xai import GP
model = GP(max_trees=100, num_generations=50)
model.fit(X_train, y_train)
model.predict(X_test)
```

## Cite

This work was published in GECCO 2019: https://dl.acm.org/citation.cfm?id=3321707.3321726

To cite, please use the following

```
@inproceedings{evans2019s,
  title={What's inside the black-box?: a genetic programming method for interpreting complex machine learning models},
  author={Evans, Benjamin P and Xue, Bing and Zhang, Mengjie},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  pages={1012--1020},
  year={2019},
  organization={ACM}
}
```
