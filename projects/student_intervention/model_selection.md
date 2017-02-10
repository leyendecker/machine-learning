# Model Evaluation

## Candidates

+ Gaussian Naive Bayes (GaussianNB)
- Decision Trees
+ Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
- K-Nearest Neighbors (KNeighbors)
- Stochastic Gradient Descent (SGDC)
+ Support Vector Machines (SVM)
- Logistic Regression

## Questions

1. Describe one real-world application in industry where the model can be applied. (You may need to do a small bit of research for this — give references!)
2. What are the strengths of the model; when does it perform well?
3. What are the weaknesses of the model; when does it perform poorly?
4. What makes this model a good candidate for the problem, given what you know about the data?


# Selection

## Gaussian Naive Bayes

1. This algorithm can be used to classify text like emails or news and predicting if it's a spam message or the category of the editorial text such as sports, politics or economics. [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.5542&rep=rep1&type=pdf]

2. It's a strong classification algorithm and can outperform more complex algorithms. It's also very fast to run and can be trained with a small dataset. It's based on the naive assumption that all variables are independent from each other so it won't fall on the curse of dimensionality.

3. The naive assumption is cited as the most common problem with this algorithm as it won't capture the relationships across 2 or more features.

4. Considering this dataset were we have a lot of features (dimensions) and not that many samples, Naive Bayes looks promising. Also, the Naive Bayes has some limitation with continuous variables that must be put into buckets, and since the dataset have just a few of those, it might be another advantage.

## Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)

1. Random Forest Classifiers can tackle a great range of Machine Learning problems. According to Breiman (2001), medical diagnosis and document retrieval problems have the many input features and each one of them contains a small amount of information and Random Forest classifiers a single tree classifier will not perform very well in this scenario but a Random Forest can result in a greater accuracy. [https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf]

2. This classifier can outperform others on a dataset with a large number of features and will not overfit opposed to single tree algorithm.

3. Although it can be used for regression, it might not performa as well as other ensemble methods like AdaBoost.

4. Ensemble methods are good candidates as they minimize the overfitting problems and usually generalize well. Random Forest is a great classifier and known to be versatile outperforming other algorithms in a large range of Machine Learning problems.


## Support Vector Machines

1. Support Vector Machines has a lot of real world application such as in bioinformatics, text mining, face recognition and image processing. (Wang, 2005) [http://www3.ntu.edu.sg/home/elpwang/pdf_web/05_svm_basic.pdf]

2. It's effective in high dimensional spaces and very versatile as kernel functions can be changed to better fit according to the domain.

3. If the number of feature are much larger than the samples it might not perform well.

4. SVM is a good candidate for this project because it's known to perform well on a large number of features.


# Choosing the Best Model

## Question 3 - Choosing the Best Model
*Based on the experiments you performed earlier, in one to two paragraphs, explain to the board of supervisors what single model you chose as the best model. Which model is generally the most appropriate based on the available data, limited resources, cost, and performance?*

Tests were made considering three machine learning models, Gaussian Naive Bayes, Random Forest and Support Vector Machine with sample size of 100, 200 and 300 instances. Considering classification performance as the F1 scores on test sample, Random Forest algorithm outperformed both SVM and Naive Bayes with the only exception on 100 samples where the SVM output a better score.
Regarding training and query time, the Naive Bayes has the best results but taking into account that the worst training time was 31 ms on the Random Forest algorithm it will not have a bad impact on the overall solution.
The results from this experiment made it very clear that Naive Bayse is not a very good solution and that SVM will perform better with smaller training set witch is not the case. The algorithm that should be used is the Random Forest.
