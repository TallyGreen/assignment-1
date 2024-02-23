# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)
import utils as u
import new_utils as nu

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)
        
        answer = {}
        answer["nb_classes_train"] = len(np.unique(ytrain))
        answer["nb_classes_test"] = len(np.unique(ytest))
        answer["class_count_train"] = np.bincount(ytrain)
        answer["class_count_test"] = np.bincount(ytest)
        answer["length_Xtrain"] = len(Xtrain)
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = Xtrain.max()
        answer["max_Xtest"] = Xtest.max()
        

        # Enter your code and fill the `answer`` dictionary

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set

        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary

        Xtrain = Xtest = np.zeros([1, 1], dtype="float")
        ytrain = ytest = np.zeros([1], dtype="int")

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        """ """
        # Enter your code and fill the `answer`` dictionary
        answer = {}
        train_list = ntrain_list
        test_list = ntest_list
        for i in range(0,len(train_list)):
            train_val = train_list[i]
            test_val= test_list[i]
            Xtrain = X[:train_val]
            ytrain = y[:train_val]
            Xtest = Xtest[:test_val]
            ytest = ytest[:test_val]

            partC = {}
            dt_clf=DecisionTreeClassifier(random_state=self.seed)
            K_cv=KFold(n_splits=5,shuffle=True,random_state=self.seed)
            C_results=u.train_simple_classifier_with_cv(Xtrain=Xtrain,ytrain=ytrain, clf=dt_clf, cv=K_cv)
            
            partC["clf"] = dt_clf 
            partC["cv"] = K_cv  
            C_scores={}
            C_scores['mean_fit_time']=partC_results['fit_time'].mean()
            C_scores['std_fit_time']=partC_results['fit_time'].std()
            C_scores['mean_accuracy']=partC_results['test_score'].mean()
            C_scores['std_accuracy']=partC_results['test_score'].std()
                    
            partC = {}
            dt_clf=DecisionTreeClassifier(random_state=self.seed)
            K_cv=KFold(n_splits=5,shuffle=True,random_state=self.seed)
            partC_results=u.train_simple_classifier_with_cv(Xtrain=Xtrain,ytrain=ytrain, clf=dt_clf, cv=K_cv)
            
            partC["clf"] = dt_clf 
            partC["cv"] = K_cv  
            partC_scores={}
            partC_scores['mean_fit_time']=partC_results['fit_time'].mean()
            partC_scores['std_fit_time']=partC_results['fit_time'].std()
            partC_scores['mean_accuracy']=partC_results['test_score'].mean()
            partC_scores['std_accuracy']=partC_results['test_score'].std()
                    
            partC["scores"] = partC_scores
    
            partD = {}
            Sh_cv=ShuffleSplit(n_splits=5,random_state=self.seed)
            partD_results=u.train_simple_classifier_with_cv(Xtrain=Xtrain,ytrain=ytrain,
                                              clf=dt_clf,
                                              cv=Sh_cv)
    
            partD["clf"] = dt_clf
            partD["cv"] = Sh_cv
            
            partD_scores={}
            partD_scores['mean_fit_time']=partD_results['fit_time'].mean()
            partD_scores['std_fit_time']=partD_results['fit_time'].std()
            partD_scores['mean_accuracy']=partD_results['test_score'].mean()
            partD_scores['std_accuracy']=partD_results['test_score'].std()
            
            partD["scores"] = partD_scores
    
    
            partF={}
            
            clf_LR=LogisticRegression(max_iter=100, multi_class='ovr', solver='lbfgs')
            
            partF_results=u.train_simple_classifier_with_cv(Xtrain=Xtrain,ytrain=ytrain, clf=clf_LR, cv=Sh_cv)
    
            partF_scores={}
            partF_scores['mean_fit_time']=partF_results['fit_time'].mean()
            partF_scores['std_fit_time']=partF_results['fit_time'].std()
            partF_scores['mean_accuracy']=partF_results['test_score'].mean()
            partF_scores['std_accuracy']=partF_results['test_score'].std()
            
    
            partF["clf_LR"] = clf_LR
            partF["clf_DT"] = dt_clf
            partF["cv"] = Sh_cv
            partF["scores_LR"] = partF_scores
    
            if partF_scores['mean_accuracy'] > partD_scores['mean_accuracy']:
                partF["model_highest_accuracy"]='Logistic Regression'
            else:
                partF["model_highest_accuracy"]='Decision Tree'
           
            partF["model_lowest_variance"]=min(partF_scores['std_accuracy']**2,partD_scores['std_accuracy']**2)
    
            partF["model_fastest"]=min(partF_scores['mean_fit_time'], partD_scores['mean_fit_time'])
        
            answer[train_list[i]] = {}
            answer[train_list[i]]["partC"] = partC
            answer[train_list[i]]["partD"] = partD
            answer[train_list[i]]["partF"] = partF
            answer[train_list[i]]["ntrain"] = train_list[i]
            answer[train_list[i]]["ntest"] = test_list[i]
            answer[train_list[i]]["class_count_train"] = list(np.bincount(ytrain))
            answer[train_list[i]]["class_count_test"] = list(np.bincount(ytest))
        """
        `answer` is a dictionary with the following keys:
           - 1000, 5000, 10000: each key is the number of training samples

           answer[k] is itself a dictionary with the following keys
            - "partC": dictionary returned by partC section 1
            - "partD": dictionary returned by partD section 1
            - "partF": dictionary returned by partF section 1
            - "ntrain": number of training samples
            - "ntest": number of test samples
            - "class_count_train": number of elements in each class in
                               the training set (a list, not a numpy array)
            - "class_count_test": number of elements in each class in
                               the training set (a list, not a numpy array)
        """

        return answer
