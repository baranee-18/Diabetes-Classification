# Diabetes-Classification

The dataset which is being used is a sample of the population of women who are at least 21 years old and belong to the Pima Indian Heritage and who are lining in and around Phoenix, Arizona, USA. They were initially tested with the criteria set by the World Health Organization. The data was collected by the National Institute of Diabetes and Digestive and Kidney Diseases. Each record in the dataset belongs to each patient and the task is to predict whether the patient is vulnerable to diabetes in future. 

Recently there has been lot of discussion about “Ensemble Learning” in which we combine models which do better than random and come up with an ensemble of such models which does way better than any single model alone. One more important thing is that the small size of the dataset will limit the performance of some algorithms. 

The basic models like Decision Tree, KNN and Naïve Bayes all had performed poorly as the accuracy for them was only around the range of 65~68% accuracy but whereas the Ensemble methods have done considerably well. This can be because the ensemble methods combine and make use of various models to come up with an ensemble of models which perform better than the individual one. The 10-fold cross validation was used and it has helped the models to learn better because rather than learning once and testing it this method will repeat the process 10 times which in return gives the model to learn more and capture more variance when compared to usual single pass. 
 
The Summarized table of Accuracy is as follows:
Ensemble algorithms and Accuracy :  
Boosting       -     75.08% 
Bagging        -     68.35% 
Stacking       -     73.8%
