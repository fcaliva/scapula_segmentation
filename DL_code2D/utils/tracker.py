import numpy as np
import pdb
class tracker(object):
    def __init__(self,num_classes):
        self.num_classes = num_classes
        self.stored_loss = np.empty((1))
        self.stored_score = np.empty((1))
        self.stored_classes_score = np.empty((1,num_classes))
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
