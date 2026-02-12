#model training, evaluation, model selection

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier #multi-layer perceptron classifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score , roc_auc_score
from sklearn.metrics import classification_report,confusion_matrix
# from train_test import split_function

class detectionmodel:
    def __init__(self,model_type = "mlp"):
        self.model_type = model_type
        self.model = self.create_model()

    def create_model(self):
        
        if self.model_type == "naive_bayes":
            return MultinomialNB(alpha=1.0)
        
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=30,
                class_weight='balanced',
                n_jobs=-1
            )
        
        elif self.model_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(128,64,32),
                activation='relu',
                learning_rate='adaptive',
                max_iter=200,
                random_state=42
            )
        else:
            raise ValueError(f"unknown model type :{self.model_type}")
        

    def training(self,X_train,y_train):
        print(f"training {self.model_type} model..")
        self.model.fit(X_train,y_train)
        print(f"{self.model_type} training completed..")

    def predict(self , X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self,X_test):
        return self.model.predict_proba(X_test)
    def evaluate_model(self,y_true,y_pred,y_pred_proba=None):
        metrics = {}

        metrics['accuracy'] = accuracy_score(y_true,y_pred)
        metrics['precision'] = precision_score(y_true,y_pred)
        metrics['recall'] = recall_score(y_true,y_pred)
        metrics['f1_score'] = f1_score(y_true,y_pred)

        #confusion matrix
        tn,fp,fn,tp = confusion_matrix(y_true,y_pred).ravel()
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['true_positive'] = tp

        #ROC - AUC
        if y_pred_proba is not None :
            metrics['roc_auc'] = roc_auc_score(y_true,y_pred_proba[:,1])
        return metrics
    
        
    def print_evaluation(self,metrics):
        print("MODEL EVALUATION RESULTS")
        print(f"\nAccuracy:{metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"\nPrecision:{metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"\nRecall :{metrics['recall']:.4f}({metrics['recall']*100:.2f}%)")
        print(f"\nF1-Score: {metrics['f1_score']:.4f}")

        if 'roc_auc' in metrics :
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

        print("\nConfusion Matrix: ")
        print(f"\nTrue Negatives(Real jobs correctly identified as jobs): {metrics['true_negative']}")
        print(f"\nFalse Positives(Real jobs incorrectly identified as fake): {metrics['false_positive']}")
        print(f"\nFalse Negatives(Fake jobs incorrectly identified as real): {metrics['false_negative']}")
        print(f"\nTrue Positives(Fake jobs correctly identified as fake): {metrics['true_positive']}")
