# XAI
Genetic programming method for explaining complex ensembles

## Overview
Interpreting state-of-the-art machine learning algorithms can be difficult. For example, why does a complex ensemble predict a particular class? Existing approaches to interpretable machine learning tend to be either local in their explanations, apply only to a particular algorithm, or overly complex in their global explanations. In this work, we propose a global model extraction method which uses multi-objective genetic programming to construct accurate, simplistic and model-agnostic representations of complex black-box estimators. We found the resulting representations are far simpler than existing approaches while providing comparable reconstructive performance. This is demonstrated on a range of datasets, by approximating the knowledge of complex black-box models such as 200 layer neural networks and ensembles of 500 trees, with a single tree.


## How to use

The GP explainer/surrogate model is implemented as sklearn classifier, and as such uses the sklearn api. To use, simply train the surrogate model on the predictions of a complex black box model. 

```python
from src.xai import GP

# Train and predict with blackbox
blackbox = ComplexModel()
blackbox.train(X_train, Y_train)
predictions = blackbox.predict(X_train)

# Use GP to make an approximation of the blackbox predictions
explainer = GP(max_trees=100, num_generations=50)
explainer.fit(X_train, predictions)

# Save our approximations
explainer.plot("model.png") 
explainer.plot_pareto("frontier.png")
```

### Example Output

A resulting model recreating the predictions from a 200 layer neural network

<p align="center">
<img src="https://i.imgur.com/e33h81y.png" width="300">
</p>

And visualising a frontier

<p align="center">
<img src="https://i.imgur.com/tZ4uU02.png" width="300">
</p>


## Evolutionary Process

![Process](https://i.imgur.com/DgMXYJn.png)

The overall training algorithm is given above. The black-box classifier is trained once only on the original data (X_train and Y_train values), then the evolutionary process is performed based on the resulting predictions (predictions) from this black-box model. The evolutionary algorithm never sees the original labels (Y_train), as this is instead attempting to recreate the predicted labels (predictions).  At the end of the evolutionary run, the result is a set of Pareto optimal models/trees which approximate the complex black-box model. Only the model with the highest reconstructive ability is used here (i.e. the largest f1). The overall evolutionary process is similar to NSGAII. When selecting individuals, the non-dominated sorting in NSGA-II algorithm is used to rank the individuals. 

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
