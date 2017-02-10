### Question 4 - Model in Layman's Terms
*In one to two paragraphs, explain to the board of directors in layman's terms how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical or technical jargon, such as describing equations or discussing the algorithm implementation.*

A single decision tree can be used as a classifier that will predict if a student will fail to graduate or not based on his/her information. Each node of the tree will test a feature from the student data and the result will split according to the test result to lower level node. The probability of a given student graduate or not should be different on a child node from it's parent. This process goes on from the root of the tree to it's leafs. The leafs of the tree will have the labels that in this case will be an yes or no answer (pass or fail). Decision Trees are very useful on Machine Learning but they have some pitfalls. One of them is that the can easily overfit meaning that the train algorithm will fit the training data but it won't generalize well.

The Random Forest Classifier algorithm will create several decision trees at random each tree will use a subset of the features at hand and the label will be given by averaging the result of each tree. By doing that this classifier performs better than the regular decision tree and also is less likely to overfitting issues.


### Question 5 - Final F<sub>1</sub> Score
*What is the final model's F<sub>1</sub> score for training and testing? How does that score compare to the untuned model?*

The final F<sub>1</sub> score if **0.9360** for training and **0.9365** for testing. It's a good improvement for the testing F<sub>1</sub> score that was **0.8983** before tuning but training score dropped from **0.9925**. This tells us that after tuning the classifier does better on generalizing results opposed to overfit the data (high difference from testing to training scores).
