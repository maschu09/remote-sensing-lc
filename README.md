# Manual

To run both notebooks, no further modules have to be installed away from the ones that were already given. Just run all cells in their given order. 

# Question Catalogue from the RF Classifier Notebook

In this section, the answers for the questions from the random forest (later often referred to as RF) classifier notebook are given. 

## Question: What would you need to do to extract only the green and the infrared channel from this data?

Each 28 x 28 = 784 values of a row/sample represent one channel, e.g. R, G, B, or NIR. Assuming that green is the second channel, we would need to only keep or pick the columns 783 to 1567 of the dataframe to extract the green channel. Note that we have to shift by one as dataframe indices start by 0 instead of 1. The same applies for the NIR channel, we are assuming to be the last one. Here we would need to pick the columns 2351 up to 3134. 

## Question: What is the advantage of this encoding compared to a simple class label like '0', '1', '2', '3', '4', '5', or text labels like 'building', 'barren_land', ...?

Some ML algorithms require that all features are in a numerical format and can not directly process a label like 'building'. Thereby, we need a way of converting categorical data into numerical. This could be done by simply encode it as an integer/digit, e.g. 0, 1, 2, 3, and so on. This encoding or representation of the classes results in a certain degree of order, which may not be desirable or may even be disadvantageous to ML. In case of categorical variables where no such ordinal relationship exists, a one-hot encoding can be applied that does not imply any form of ordinal structure or distance between two categories or classes. 

## Question: Why use 'extend' here and 'append' above?

The key difference between both methods is that extend() adds all elements of a list individually to a list, while append() adds the whole list as a single element into the list. In the second code block, where we use extend, 'train_idx' shall be a simple single 1-dimensional list. In the first code block, where we use append, we want a list of lists. For each individual class we want its own list which are all stored in one superior list 'sample_idx'.

A short code example to illustrate the differences:

```
test_list.extend([5, 8, 12])
print(test_list)
[5, 8, 12]

test_list.append([5, 8, 12])
print(test_list)
[[5, 8, 12]]
```

## Question: What is wrong with the above code?

```
num_train = 1000
num_test = 100
train_idx = []
test_idx = []
for column in column_names:
    # find all indices of a given class
    class_idx = labels_df[column] == 1
    # randomly select num_train and num_test values from this index list - make sure to avoid duplicates
    train_idx.extend(np.random.choice(np.where(class_idx.values)[0], size=num_train, replace=False).tolist())
    test_idx.extend(np.random.choice(np.where(class_idx.values)[0], size=num_test, replace=False).tolist())
print(f'number of train indices: {len(train_idx)}, number of test indices: {len(test_idx)}')
```

The problem with the code above is that it is possible for a data point to be included in both the training and test dataset. Thereby, our model potentially already know (some) data of the testset and thus perform better than it might actually should. This would induce what we also call data leakage as we have directly "corrupted" that data as valid datasets. We can not assure that the testset performance is a reliable measurement of generalisation of our model.


## Question: Why do you want to shuffle the samples in the train and test datasets?

We just want to make sure that the model is not learning anything from the order of the data/samples. Maybe all samples of one class are grouped together or we have spatial ordering which could potential lead to biased training and thereby unreliable evalution of the models performance. Shuffle the samples prevents this issues and results in better generalization.

# RF Classifier Tasks

In the following section, additional information and demanded answers regarding the tasks 2 and 3 are given.  

## Task 2.3 - How does the modification of using only RGB channels affect the classification accuracy?

In the case where we only make use of the channels r, g, and b and drop the NIR channel, the overall classification accuracy is going down. The classifier achieve roughly 5 % less overall accuracy than with all four channels. When taking a closer look at the specific per-class accuracies, it can be observed that an overall decrease in accuracy is the case for all classes. But I think it is worth mentioning that this is not equally for all classes. The class 'barren_land' and 'trees' were affected the most and suffered a decrease in per-class accuracy around 10 %. The others classes accuracy only decreased by around 1-3 %. 

## Task 2.4 - Make use of only R, G, and NIR. What changes?

In this second case where we only make use of the channels r, g, and nir and drop the b channel, the overall classification accuracy is slightly below our baseline using all channels. The classifier is roughly only 1-2 % less accurate than before. Like in the first case, when taking a closer look at the specific per-class accuracies, it can be observed that an overall decrease in accuracy is not the case for all classes. This time, most classes accuracies decrease relatively equally around 1 % or remained the same. But there are two expections in the classes 'road' as well as 'water'. The class 'road' decreased by 4 % while 'water' even increased slightly.

## Task 2.5 - Justify your choice of the hyper parameter and of the new value. What is your expectation? Will the classification accuracy increase or decrease? Will the model become more robust or less robust?  Document your results. Do the results agree with your expectation?

For this task I picked the 'max_depth' as hyperparameter to modify. This hyperparameter denotes to the maximal depth of the trees and can be roughly described as the complexity of an individual tree (how many splits we want to allow). As default, the max_depth is set to 'None' which results in trees where all leaves are either pure or contrain less than 'min_samples_split' samples. 

To better observe the effect of modifying the value, I want to test several values for the max_depth ranging from 1 up to 50. From 1 to 100 I go up in increments of five. In terms of expected behaviour when changing this hyperparameter with increasing depth, I assume that we will observe a rise in accuracy in the beginning. At some point, the classifier will reach its peak accuracy and will ongoing decrease. The rise in the beginnng due to underfitting and decrease after the peak due to overfitting. Thereby, underfitting describes the phenomenon when we have an excessively simplistic formulation of the model. E.g., trees with only one split will not be sufficient to properly describe a set of rules to determine what type of landcover is shown. On the other hand, overfitting occurs when we have an excessively flexible and complex formulation. In this case, the model is more like memorizing the training data (including noise) and generalizes bad to the new unseen test data.

To summarize, increasing the max_depth will result in more complex trees and thus will be able to capture more complex relationships. But we have to carefully select this hyperparameter as it will make our classifier less robust and generalize less well if we set it to big. As small addition, this way of observing how the choice of the hyperparameter affects the performance on the testset would not be a smart decision as we introduce data leakage and corrupt our testset once we looked into it. Thereby, a triple split into train-, validation-, and testset would be more suitable.  

When the changes are implemented, the behaviour observed is exactly as expected. The initial steps in particular have a huge impact on the classifier’s accuracy, but their influence gradually diminishes. Around a max_depth of 50 to 100, the accuracy is even going down slightly.  

# CNN Classifier Tasks

## Task 3.2 - Document what you need to change from task 2.2.

While the structure of both code to evaluate the performance of the classifiers is relatively similar, there are some key differences. 

Starting of the format of the prediction we get from the different classifier. In case of the RF, we get 2-dimensional list that stores the one-hot encoded class prediction. This has to be extracted into class labels for computing the per-class accuracy. On the other hand, the CNN directly gives us the predicted labels as integers stored in 'pred_labels'. 

The main difference arises when plotting randomly selected correctly and misclassified samples. While in the case of the RF case we have to reconstruct the image from a flat vector before visualization. In contrast, the CNN case requires different handling due to tensor-based image storage and normalization. The images are stored as tensors in an array. Thereby, no manual reshaping is required. 
However, we have to denormalize the image data as our CNN was trained with normalized inputs. Therefore, for visualization purposes, this normalization must be reversed by multiplying the data with the standard deviation and adding the mean of the dataset. To ensure we do not run into any errors or warning, the resulting image data is clipped to the range of 0 to 255. Otherwise it could be possible that we encounter invalid values after floating-point operations. 
In the next step, we transpose our image data in tensor format as PyTorch and Matplotlib are using and expecting different shapes. 
Lastly, we drop the NIR channel as Matplotlib can only work with the RGB channels. Because our image data is in float format after denormalization, Matplotlib expect and interprets it in the [0,1] range. Therefore we have to divide them by 255.             

## Task 3.3 - How do the results degrade? Is the effect smaller or larger than with the random forest?

The results are very similar to the ones with the random forest classifier. It can be observed that the overall accuracy of the CNN classifier is also decreasing slightly. When taking a closer look at the per-class accuracies, we can observe a similar behaviour of differences between per-class accuricies. For example, the accuracy for the class 'trees' dropped around 14 % while 'building' only decreased by roughly 4 %. Furhter, classes like 'grassland' and 'road' benefitted by the changes and increased by roughly 8 % in accuracy. This behaviour of improvements in certain per-class accuracies was not present with the RF classifier. 

## Task 3.4 - Justify your choice of the hyper parameter and of the new value. What is your expectation? Will the classification accuracy increase or decrease? Will the model become more robust or less robust? Implement the change and document your results. Do the results agree with your expectation?

For this task, I have chosen the number of epochs as hyperparameter of the training that I want to modify. The number of epochs refers to the number of complete passes through the training dataset during the training of the CNN classifier. In simple words, this means how long our CNN classifier is learning on the given data.  

This leads to the expectation that the accuracy of the CNN classifier will improve as the number of epochs increases. In particular, the training loss should decrease and the accuracy should improve. As it was the case within task 2.5 on the RF classifier, this potentially lead to overfitting and the classifier learning the training data instead of the underlying pattern we actually want to learn. Thus resulting in worse performance on the testset and bda generalisation capabilities.

When implemented the modifications and execute the code, the exact same behaviour as expected can be observed. As with the randomforest classifier, the first increment steps of the hyperparameter have the biggest impact on the accuracy while it gradually diminishes. But in this case the expected overfitting is not that present. Further, it can also be observed that the per-class accuracy is imporving relatively equal among all classes. There is not a single class where the per-classs accuracy is below the baseline we have seen from the first trained CNN in this notebook.

## Author

Nils Hornstein

