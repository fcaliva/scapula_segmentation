import numpy as np
import pdb
class tracker(object):
    def __init__(self,num_classes):
        self.num_classes = num_classes
        self.stored_loss = np.empty((1))
        self.stored_score = np.empty((1))
        self.stored_classes_score = np.empty((1,num_classes))
        self.stored_names = np.empty(1)
        self.stored_labels = np.empty((1,num_classes))
        self.stored_predictions = np.empty((1,num_classes))
        self.cum_loss = 0.0
        self.cum_score = 0.0
        self.classes_cum_score = np.zeros((1,num_classes))
        self.iter = 0

    def increment(self,loss,score,class_score):
        if self.iter > 0:
            self.stored_loss          = np.append(self.stored_loss, loss)
            self.stored_score         = np.append(self.stored_score, score)
            self.stored_classes_score = np.vstack([self.stored_classes_score,class_score])
        else:
            self.stored_loss          = loss
            self.stored_score         = score
            self.stored_classes_score = class_score

        self.cum_loss += loss
        self.cum_score += score
        self.classes_cum_score += class_score
        self.iter += 1

    def average(self):
        return self.cum_loss/self.iter, self.cum_score/self.iter, self.classes_cum_score/self.iter

    def stdev(self):

        if self.stored_classes_score.ndim == 1:
            std_classes = np.zeros((1,self.num_classes))
        else:
            std_classes =  np.std(self.stored_classes_score,axis=0)
        return np.std(self.stored_loss), np.std(self.stored_score), std_classes

    def store_data(self,name_to_store,label_to_store,pred_to_store):
        if self.iter > 0:
            self.stored_names = np.vstack((self.stored_names,name_to_store))
            self.stored_labels = np.vstack((self.stored_labels,label_to_store))
            self.stored_predictions = np.vstack((self.stored_predictions,pred_to_store))
        else:
            self.stored_names = name_to_store
            self.stored_labels = label_to_store
            self.stored_predictions = pred_to_store

    def roc(self):
        from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
        from matplotlib import pyplot as plt

        if self.stored_labels.shape[-1]==1:
            threshold = 0.5
            gt = self.stored_labels.astype('float32')>0.5
            pred = self.stored_predictions.astype('float32')>0.5

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            fpr, tpr, _ = roc_curve(gt,pred,pos_label=0)

        else:
            gt = np.argmax(self.stored_labels,axis=1)
            pred = np.argmax(self.stored_predictions,axis=1)

            cm = confusion_matrix(gt, pred)
            print('confusion_matrix:\n {}'.format(cm))
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(self.num_classes):
                fpr[i], tpr[i], _ = np.round(roc_curve(gt==i, pred == i),4)
                roc_auc[i] = np.round(auc(fpr[i], tpr[i]),4)
        print(classification_report(pred,gt, digits=4))
        print('roc_auc: {}'.format(roc_auc))
