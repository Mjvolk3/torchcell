---
id: 8idydyfk79r59r2c1rualwy
title: 'Sahand-Shafeei.preliminary-project-outline.2025.05.05'
desc: ''
updated: 1746811155367
created: 1746499667043
---

## Hypothesis Driving the Work

Using more structure as a prior will help improve inference when we have few experiments over a specific metabolic pathway.

## Experiment Outline

1. Benchmark with machine learning models that don't use structures.

| gene A  | gene B  | ...     | GFP MFI (supervised label) |
|:--------|:--------|:--------|:---------------------------|
| [float] | [float] | [float] | [float]                    |

This table has $306$ rows. Looks like $307$ but one row looks empty. Double checked and looks like that is only missing error.  $250/7,7764$. $6^5 = 7,776 \text{ total combinations}$. From looking over the SI pdf's there are columns with build signifying whether certain strains are in the larger table. I counted total of $43$. This is mismatched from $307-43=264 \neq 250$.  This must mean there are something like $14$ duplicates or controls. The tables in SI are not supplied as `csv` or `xlsx`. These will have to be converted to csv or xlsx and read in. Once you have all data read in you will be able to tell where the different data comes from. Some of the suggestions that aren't marked as build might be in the table. The goal is to reduce the table to the $250$ original designs. You try to determine where the data comes from otherwise the comparisons will be meaningless. Once you have sorted out test and train. You can can move onto model training and testing.

I have looked and You can use some standard tabular machine learning models to run supervised benchmarking: Random Forest, SVR, XGBoost, MLP, etc. You are doing a supervised regression. Since there are so few combinations you should be able to run inference over all combinations. Once models are trained you can do a direct comparison based on their predictions on test and your predictions on test. Then you can test your predictions versus their predictions using rankings. Compute Kendall Tau which measures concordances of two lists. $P_{\mathrm{agree}}=\frac{\tau+1}{2}$. After computing Kendall Tau you can help others understand this saying "If you select any two number at random there is a $P_{\text{agree}}$ that the predicted ordering is correct." Also compare with other regression metrics: RMSE, MAE, Pearson, Spearman.

Running inference over all $7,776$ will help you make comparisons for strains that are not built but still have predictions. We want to be able to see if these algorithms perform better or worse than ART which has no context. These model is decently strong as it is an ensemble of many tabular learners.

2. Do the exact same thing as 1. but using TabPFN. The hypothesis here is that providing some pretraining based on causal structure will allow for better predictions and there for improved design build test learn (DBTL) cycles. This also will help support testing the hypothesis that more refined and specific structure will help improve predictions even more.

3. Compare the TabPFN stochastic causal block model to the metabolic graph. Their similarity should help us interpret and speculate why the method does or does not work.

## Future Experiments

1. Rerun the experiments using more structure specific to the cell.

- If TabPFN doesn't work, maybe the structure isn't specific enough and structure from the cell can help.
- If TabPFN does work, maybe the structure is sufficient metabolic inference. Proving that extra structure isn't necessary will be useful for simplifying the metabolic engineering workflow.

Regardless of the result, the outcome will guide our next steps and help guide us in developing machine learning methods for metabolic engineering.

## Rough Presentation Outline

Slides (Not including title):

- 1-8 : Introduction
  - 1-4 : Metabolic Engineering - Goals, methods, difficulties
  - 5-6 : Machine learning in metabolic engineering DBTL
  - 7-8 : Adding "context" to help improve models - TabPFN, GEMs + Neural networks
- 9 - 12 : Tryptophan Study
  - 9 - 10: Describe experimental setup
  - 11 - 12: Describe data
- 13 : Outline Project aims
  *These need some work*
  - 1. Show how much context is needed for tryptophan production. None, generic, specific.
  - 2. Generalize to other metabolic engineering projects. (see difficult with dodecanol production in original ART paper).
  - 3. Design a model that can make inferences across *S. cerevisiae* metabolism with low $n$.
- 14 - 15 : Benchmark Models
- 16 - 17 : TabPFN
- 18 : Compare TabPFN structural causal model to metabolic network.
- 19 : Discussion of results and future work.
- 20 : Conclusion

## 2025.05.06 Meeting

- We want to Compare TabPFN DAG structure to the metabolic structure this is a longer term goal that we don't need to worry about now but can be discussed as future work.
- The comparisons of our inference model with their model will be imperfect since we cannot generate our own data and run experiments to get measurements but it is the best we can do. We are focused on comparing supervised learning models, not acquisition function. You cannot compare the acquisition function without running more experiments
- Put the additional table into their own excel spreadsheets and give them proper names according to the SI. We want all of the data from there. You should do the same for the two methods that don't use build. We can still compare against their predictions. 4 total tables.

