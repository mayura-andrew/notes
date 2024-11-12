
# What is Machine Learning

In basic of terms, ML is the process of training a piece of software, called a model, to make useful predictions or generate content from data.

## modal

is a mathematical relational relationship derived from data that an ML system uses to make predictions

## Type of ML systems 

- Supervised learning 
- Unsupervised learning
- Reinforcement learning 
- Generative AI

### Supervised learning 

 supervised learning models can make predictions after seeing lot a of data  with the correct answers and then discovering the connections between the elements in the data that produce the correct answers. 

Ex - This is like a student learning new material by studying old exams that contain both questions and answers. Once the student has trained on enough old exams, the student is well prepared to take a new exam. These ML systems are "supervised" in the sense that a human gives the ML system data with the known correct results.

Two of the most common use cases

1. Regression - A regression model predicts a numeric value.

2. Classification - predict the likelihood that something belongs to a category. like  boolean expression (if else?) 
			Two main groups,
				- Binary Classification ( from only two values) ( yes or no)
				- Multi-class Classification (from more than two values)

### Unsupervised Learning 

Models make predictions by being given data that does not contain any correct answers.  
An unsupervised learning model's goal is to identify meaningful patterns among the data. 

A commonly used unsupervised learning model employs a technique called #clustering.

The model finds data points that demarcate natural groupings.
![[Pasted image 20241107134018.png]]

**Figure 1**. An ML model clustering similar data points.


![[Pasted image 20241107134002.png]]
**Figure 2**. Groups of clusters with natural demarcations.

Clustering differs from classification because the categories aren't defined by you.


![[Pasted image 20241107134432.png]]

**Figure 3**. An ML model clustering similar weather patterns.


![[Pasted image 20241107134455.png]]
**Figure 4**. Clusters of weather patterns labeled as snow, sleet, rain, and no rain.



A supervised approach is given data that contains the correct answer. The model's job is to find connections in the data that produce the correct answer. An unsupervised approach is given data without the correct answer. Its job is to find groupings in the data.

### Reinforcement Learning

Models make predictions by getting rewards or penalties based on actions performed within an environment. 

A reinforcement learning is used to train robots to perform tasks, like walking around a room, and software programs like AlphaGo to play the game of Go.

### Generative AI

is a class of models that creates content from user input. 

type of input to type of output. 
		
		Text-to-text
		Text-to-image
		Text-to-video
		Text-to-code
		Text-to-speech
		Image and text-to-image

To produce unique and creative outputs, generative models are initially trained using an unsupervised approach, where the model learns to mimic the data it's trained on.  

The model is sometimes trained further using supervised or reinforcement learning on specific data related to tasks the model might be asked to perform, for example, summarize an article or edit a photo.


# 1. Supervised Learning 

## Foundational supervised learning concepts

There are few core concepts

- Data
- Model
- Training
- Evaluating
- Inference


#### Data 

two main categories 
		- Labeled - A labeled example consists of one or more features and a label. Labeled examples are used during training.
		- Unlabeled - Unlabeled examples are used during inference.
A data set is characterized by its ==size (number of examples)== and ==diversity(the range those examples cover)==. 
Good datasets are both large and highly diverse.

Label - is the "answer" or the value we want the model to predict. 


A large number of examples that cover a variety of use cases is essential for a machine leaning systems to understand the underling patterns in the data. a model trained on this type of dataset is more likely to make good predictions on new data. 

==(Large Size / High diversity)==


#### Model

In supervised learning, a model is the complex collection of numbers that define the mathematical relationship from specific input feature patterns to specific output label values. 

The model discovers these patterns through training. 


#### Training 

Before a supervised model  can make predictions, it must be trained. 

![[Pasted image 20241108123905.png]]


The model's goal is to work out the best solution for predicting the labels from the features. 

The model finds the best solution by comparing its predicted value to the label's actual value. 

Based on the difference between the predicted and actual values defined as the loss the model gradually updates its solution. 

In other words, the model learns the mathematical relationship between the features and the label so it can make the best predictions on unseen data.

![[Pasted image 20241108133008.png]]
**Figure 4**. An ML model updating its predictions for each labeled example in the training dataset.


#### Evaluating

![[Pasted image 20241108133235.png]]
Depending on the model's predictions, we might do more training and evaluating before deploying the model in a real-world application.


 If you know beforehand the value or category you want to predict, you'd use supervised learning. However, if you wanted to learn if your dataset contains any segmentations or groupings of related examples, you'd use unsupervised learning.


   ![[Pasted image 20241108151400.png]]


==Regression Predict a number infinitely many possible outputs.==

Classification predict categories small number of possible outputs.

# 2. Unsupervised Learning

==Data only comes with inputs x==, but ==not output labels y.==
Algorithm has to find structure in the data.

#Clustering -> Group similar data points together. 

#Anomaly_detection -> Find unusual data points.

#Conditionality_reduction  -> Compress data using fewer number.

example case scenarios,
 - Given a set of news articles found on the web, group them into sets of articles about the same stories.
 - Given a database of customer data, automatically discover market segments and group customers into different market segments.
 