import numpy as np
from numpy.typing import NDArray
from typing import Any
import utils as u
import new_utils as nu
from sklearn.metrics import top_k_accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,make_scorer, f1_score,accuracy_score, recall_score,precision_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold,cross_val_score

"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        print(f"{uniq=}")
        print(f"{counts=}")
        print(f"{np.sum(counts)=}")

        return {
            "class_counts": {},  # Replace with actual class counts
            "num_classes": 0,  # Replace with the actual number of classes
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """
        # Enter code and return the `answer`` dictionary
        X,y =nu.load_mnist_dataset()
        X=nu.scale(X)
         
        ntrains = [1000, 5000, 10000]
        ntests = [200, 1000, 2000]
        ks = [1, 2, 3, 4, 5]
         
        answer = {}
         
        for ntrain, ntest in zip(ntrains, ntests):
             # Assuming X and y are predefined datasets
             Xtrain, Xtest = X[:ntrain], X[ntrain:ntrain+ntest]
             ytrain, ytest = y[:ntrain], y[ntrain:ntrain+ntest]
             
             model = nu.LogisticRegression(max_iter=300, multi_class='multinomial', solver='lbfgs')
             model.fit(Xtrain, ytrain)
             
             train_scores = [top_k_accuracy_score(ytrain, model.predict_proba(Xtrain), k=k) for k in ks]
             test_scores = [top_k_accuracy_score(ytest, model.predict_proba(Xtest), k=k) for k in ks]
             '''
             plt.figure(figsize=(8, 5))
             plt.plot(ks, train_scores, label='Training', marker='o')
             plt.plot(ks, test_scores, label='Testing', marker='o')
             plt.xlabel('k')
             plt.ylabel('Top-k Accuracy')
             plt.title(f'Top-k Accuracy vs. k (ntrain={ntrain}, ntest={ntest})')
             plt.legend()
             plt.grid(True)
             plt.show()
             '''
             
             for k, score_train, score_test in zip(ks, train_scores, test_scores):
                 answer[k] = {"score_train": score_train, "score_test": score_test}
             
             answer["clf"] = model
             
             answer["plot_k_vs_score_train"] = list(zip(ks, train_scores))
             answer["plot_k_vs_score_test"] = list(zip(ks, test_scores))
             
             # Additional analysis on rate of accuracy change for testing data
             # This is a placeholder for detailed analysis
             answer["text_rate_accuracy_change"] = "when the K becomes larger, the test accuracy rate becomes higher."
             
             # Comments on the usefulness of top-k accuracy metric
             answer["text_is_topk_useful_and_why"] = "Top-k accuracy is useful to analysis data when there are multiple classifers.  " \
                                          
        

        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """

        return answer, Xtrain, ytrain, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}
        
        '''Xtrain, ytrain, Xtest, ytest=prepare_data()'''
        '''
        Xtrain, ytrain= nu.filter_out_7_9s(Xtrain, ytrain)
        Xtest, ytest= nu.filter_out_7_9s(Xtest, ytest)
        ytrain = ytrain.astype(int)
        ytrain[ytrain == 7] = 0
        ytrain[ytrain == 9] = 1
        ytest = ytest.astype(int)
        ytest[ytest == 7] = 0
        ytest[ytest == 9] = 1
        mask = (np.random.rand(len(ytrain)) < 0.1) | (ytrain == 0)
        Xtrain = Xtrain[mask]
        ytrain = ytrain[mask]
        mask = (np.random.rand(len(ytest)) < 0.1) | (ytest == 0)
        Xtest = Xtest[mask]
        ytest = ytest[mask]
        answer["length_Xtrain"] = len(Xtrain)  # Number of samples
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = Xtrain.max()
        answer["max_Xtest"] = Xtest.max()
        '''
        '''X=Xtrain
        y=ytrain'''
        Xtrain, ytrain= u.filter_out_7_9s(X, y)
        Xtest, ytest= u.filter_out_7_9s(Xtest, ytest)
        ytrain = ytrain.astype(int)
        ytrain[ytrain == 7] = 0
        ytrain[ytrain == 9] = 1
        ytest = ytest.astype(int)
        ytest[ytest == 7] = 0
        ytest[ytest == 9] = 1
        mask = (np.random.rand(len(ytrain)) < 0.1) | (ytrain == 0)
        Xtrain = Xtrain[mask]
        ytrain = ytrain[mask]
        mask = (np.random.rand(len(ytest)) < 0.1) | (ytest == 0)
        Xtest = Xtest[mask]
        ytest = ytest[mask]
        answer["length_Xtrain"] = len(Xtrain)  # Number of samples
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = Xtrain.max()
        answer["max_Xtest"] = Xtest.max()
        X=Xtrain
        y=ytrain

        # Answer is a dictionary with the same keys as part 1.B

        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""

        # Enter your code and fill the `answer` dictionary
        answer = {}

        #partB_answer,X,y,Xtest,ytest = self.partB(X=X,y=y,Xtest=Xtest,ytest=ytest)

        clf=SVC(random_state=self.seed)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        scoring = {'f1': make_scorer(f1_score, average='macro'),
           'precision': make_scorer(precision_score, average='macro'),
           'recall': make_scorer(recall_score, average='macro'),
           'accuracy':'accuracy'}
        scores_cv = {metric: cross_val_score(clf, X, y, scoring=scoring[metric], cv=cv)
          for metric in scoring}
        
        scores_cv_stra={}
        scores_cv_stra['mean_accuracy']=np.mean(scores_cv['accuracy'])
        scores_cv_stra['mean_recall']=np.mean(scores_cv['recall'])
        scores_cv_stra['mean_precision']=np.mean(scores_cv['precision'])
        scores_cv_stra['mean_f1']=np.mean(scores_cv['f1'])
        scores_cv_stra['std_accuracy']=np.std(scores_cv['accuracy'])
        scores_cv_stra['std_recall']=np.std(scores_cv['recall'])
        scores_cv_stra['std_precision']=np.std(scores_cv['precision'])
        scores_cv_stra['std_f1']=np.std(scores_cv['f1'])
        
        answer["scores"]=scores_cv_stra
        answer['cv']=cv
        answer['clf']=clf

        clf.fit(X,y)
        y_pred_train=clf.predict(X)
        y_pred_test=clf.predict(Xtest)


        if scores_cv_stra['mean_precision'] > scores_cv_stra['mean_recall']:
            answer["is_precision_higher_than_recall"]= True
        else:
            answer["is_precision_higher_than_recall"]= False

        answer['explain_is_precision_higher_than_recall']='Yes, Precision is higher than recall  suggests that the classifier is more adept at minimizing false positive errors, emphasizing its accuracy in correctly identifying positive instances.'
       

        answer['confusion_matrix_train'] = confusion_matrix(y,y_pred_train)
        answer['confusion_matrix_test']  = confusion_matrix(ytest,y_pred_test)

        
        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        
        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

        clf= SVC(random_state=self.seed, class_weight={0: class_weights[0], 1: class_weights[1]})

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        scoring = {
            'f1': make_scorer(f1_score, average='macro'),
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'accuracy': 'accuracy'
        }

        # Perform cross-validation
        scores_cv = {metric: cross_val_score(clf, X, y, scoring=scoring[metric], cv=cv)
                              for metric in scoring}

        # Calculate mean and standard deviation of scores
        scores_cv_wt = {}
        scores_cv_wt['mean_accuracy'] = np.mean(scores_cv['accuracy'])
        scores_cv_wt['mean_recall'] = np.mean(scores_cv['recall'])
        scores_cv_wt['mean_precision'] = np.mean(scores_cv['precision'])
        scores_cv_wt['mean_f1'] = np.mean(scores_cv['f1'])
        scores_cv_wt['std_accuracy'] = np.std(scores_cv['accuracy'])
        scores_cv_wt['std_recall'] = np.std(scores_cv['recall'])
        scores_cv_wt['std_precision'] = np.std(scores_cv['precision'])
        scores_cv_wt['std_f1'] = np.std(scores_cv['f1'])

        # Fit the classifier on the entire training data
        clf.fit(X, y)

        # Predict on training and testing data
        y_pred_train_wt= clf.predict(X)
        y_pred_test_wt = clf.predict(Xtest)

        # Generate confusion matrices
        confusion_matrix_train_wt = confusion_matrix(y, y_pred_train_wt)
        confusion_matrix_test_wt= confusion_matrix(ytest, y_pred_test_wt)

        answer["scores"] = scores_cv_wt
        answer['cv'] = cv
        answer['clf'] = clf
        answer['class_weights'] = class_weights
        answer['confusion_matrix_train'] = confusion_matrix_train_wt
        answer['confusion_matrix_test'] = confusion_matrix_test_wt
        answer['explain_purpose_of_class_weights'] = "Class weights are employed to mitigate class imbalance by assigning greater penalties to misclassifications of the minority class."
        answer['explain_performance_difference'] = "The performance difference attributed to class weights indicates an enhanced capability of the model to generalize to the minority class, resulting in more balanced performance metrics across all classe"

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer
