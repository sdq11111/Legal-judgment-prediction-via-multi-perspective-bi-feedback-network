# -*-coding:utf-8-*-

import numpy as np
from collections import defaultdict
import re
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import sys
from selftools import *
import os
from sklearn.utils import class_weight
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from keras import initializers
from config import *
from final_selfmodel import *
config = Config()


class Metrics(Callback):
    def cal_metric(self, name, predict, target):

        micro_precesion = precision_score(target, predict, average="micro")
        macro_precesion = precision_score(target, predict, average="macro")
        micro_recall = recall_score(target, predict, average="micro")
        macro_recall = recall_score(target, predict, average="macro")
        micro_f1 = f1_score(target, predict, average="micro")
        macro_f1 = f1_score(target, predict, average="macro")
        acc = accuracy_score(target, predict)
        score = macro_f1

        print("\n" + name + ": \n")
        print("\n acc: %f mico precesion: %f ,macro precesion: %f \n micro recall: %f ,macro recall: %f \n micro f1: %f ,macro f1: %f \n score %f \n" % (acc, micro_precesion, macro_precesion, micro_recall, macro_recall, micro_f1, macro_f1, score))

        return score

    def on_train_begin(self, logs={}):

        self.scores = []

    def on_epoch_end(self, epoch, logs={}):

        accu_pred, law_pred, term_pred = self.model.predict([self.validation_data[0], self.validation_data[1], self.validation_data[2], self.validation_data[3], self.validation_data[4], self.validation_data[5], self.validation_data[6], self.validation_data[7], self.validation_data[8], self.validation_data[9], self.validation_data[10]
            , self.validation_data[11]
            ])
#        accu_pred, law_pred, term_pred = self.model.predict(self.validation_data[0])

        accu_np = np.asarray(accu_pred)
        accu_max = accu_np.max(axis=1).reshape(-1, 1)
        accu_pred = np.floor(accu_np/accu_max)
#        print(accu_pred[0])

        law_np = np.asarray(law_pred)
        law_max = law_np.max(axis=1).reshape(-1, 1)
        law_pred = np.floor(law_np/law_max)
#        print(law_pred[0])
#        accu_pred = (np.asarray(accu_pred)).round()
#        law_pred = (np.asarray(law_pred)).round()

        term_np = np.asarray(term_pred)
        term_max = term_np.max(axis = 1).reshape(-1, 1)
        term_pred = np.floor(term_np/term_max)


        accu_target = self.validation_data[12]
        law_target = self.validation_data[13]
        term_target = self.validation_data[14]
        
        accu_score = self.cal_metric("accu", accu_pred, accu_target)
        law_score = self.cal_metric("law", law_pred, law_target)
        term_score = self.cal_metric("term", term_pred, term_target)
        score = accu_score + law_score + term_score
        print("\n Final socre  : " + str(score))

        if self.scores == [] or score > max(self.scores):
            #filepath = "finalmodel.h5"
            filepath="finalmodel_cailsmall.h5"
            print("now is saving ..")
            self.model.save(os.path.join(data_path, "generated/" + filepath))

        self.scores.append(score)

        return


def cross(y_true, y_pred):
    return y_pred

if __name__ == "__main__":
    metrics = Metrics()
    model = build_final_graph()
    #model=build_final()
    model.compile(loss={'accu_preds': 'categorical_crossentropy',
                        'law_preds': 'categorical_crossentropy',
                        'term_preds': 'categorical_crossentropy',
                        },
                  optimizer='adam',
                  )
    print(model.metrics_names)

    train_facts, train_laws, train_accus = load_data("new_data_train_cuted")
    test_facts, test_laws, test_accus = load_data("new_data_test_cuted")
    train_singlefacts = np.load(os.path.join(data_path, "generated/new_data_train_cuted_singlefact.npy"))
    test_singlefacts= np.load(os.path.join(data_path, "generated/new_data_test_cuted_singlefact.npy"))
    train_terms = np.load(os.path.join(data_path, "generated/new_data_train_cuted_term.npy"))
    test_terms = np.load(os.path.join(data_path, "generated/new_data_test_cuted_term.npy"))
    
    train_pair_front_word = np.load(os.path.join(data_path, "generated/new_data_train_pair_front_id.npy"))
    train_pair_last_word = np.load(os.path.join(data_path, "generated/new_data_train_pair_last_id.npy"))
    train_pair_front_isword = np.load(os.path.join(data_path, "generated/new_data_train_pair_front_word.npy"))
    train_pair_last_isword = np.load(os.path.join(data_path, "generated/new_data_train_pair_last_word.npy"))
    
    train_pair_front_num = np.load(os.path.join(data_path, "generated/new_data_train_pair_front_num.npy"))
    train_pair_last_num = np.load(os.path.join(data_path, "generated/new_data_train_pair_last_num.npy"))
    train_pair_front_isnum = np.load(os.path.join(data_path, "generated/new_data_train_pair_front_dig.npy"))
    train_pair_last_isnum = np.load(os.path.join(data_path, "generated/new_data_train_pair_last_dig.npy"))
    
    
    
    
    test_pair_front_word = np.load(os.path.join(data_path, "generated/new_data_test_pair_front_id.npy"))
    test_pair_last_word = np.load(os.path.join(data_path, "generated/new_data_test_pair_last_id.npy"))
    test_pair_front_isword = np.load(os.path.join(data_path, "generated/new_data_test_pair_front_word.npy"))
    test_pair_last_isword = np.load(os.path.join(data_path, "generated/new_data_test_pair_last_word.npy"))
    
    test_pair_front_num = np.load(os.path.join(data_path, "generated/new_data_test_pair_front_num.npy"))
    test_pair_last_num = np.load(os.path.join(data_path, "generated/new_data_test_pair_last_num.npy"))
    test_pair_front_isnum = np.load(os.path.join(data_path, "generated/new_data_test_pair_front_dig.npy"))
    test_pair_last_isnum = np.load(os.path.join(data_path, "generated/new_data_test_pair_last_dig.npy"))
    
    callback_lists = [metrics]
    print("model fitting - Hierachical attention network")

    train_accu_list = [list(range(0, config.num_accu_liu))] * train_facts.shape[0]
    train_law_list = [list(range(0, config.num_law_liu))] * train_facts.shape[0]
    train_term_list = [list(range(0, config.num_term_liu))] * train_facts.shape[0]
    train_accu_input = np.asarray(train_accu_list, dtype='int32')
    train_law_input = np.asarray(train_law_list, dtype='int32')
    train_term_input = np.asarray(train_term_list, dtype='int32')
    
    
    train_pair_front_word = np.asarray(train_pair_front_word, dtype = 'int32')
    train_pair_last_word = np.asarray(train_pair_last_word, dtype = 'int32')
    train_pair_front_isword = np.asarray(train_pair_front_isword, dtype = 'int32')
    train_pair_last_isword = np.asarray(train_pair_last_isword, dtype = 'int32')
    train_pair_front_num = np.asarray(train_pair_front_num, dtype = 'int32')
    train_pair_last_num = np.asarray(train_pair_last_num, dtype = 'int32')
    train_pair_front_isnum = np.asarray(train_pair_front_isnum, dtype = 'int32')
    train_pair_last_isnum = np.asarray(train_pair_last_isnum, dtype = 'int32')
    
    
    
    test_pair_front_word = np.asarray(test_pair_front_word, dtype = 'int32')
    test_pair_last_word = np.asarray(test_pair_last_word, dtype = 'int32')
    test_pair_front_isword = np.asarray(test_pair_front_isword, dtype = 'int32')
    test_pair_last_isword = np.asarray(test_pair_last_isword, dtype = 'int32')
    test_pair_front_num = np.asarray(test_pair_front_num, dtype = 'int32')
    test_pair_last_num = np.asarray(test_pair_last_num, dtype = 'int32')
    test_pair_front_isnum = np.asarray(test_pair_front_isnum, dtype = 'int32')
    test_pair_last_isnum = np.asarray(test_pair_last_isnum, dtype = 'int32')
    
    
    print (train_accu_input.shape)
    print (train_law_input.shape)
    print (train_facts.shape)

    test_accu_list = [list(range(0, config.num_accu_liu))] * test_facts.shape[0]
    test_law_list = [list(range(0, config.num_law_liu))] * test_facts.shape[0]
    test_term_list = [list(range(0, config.num_term_liu))] * test_facts.shape[0]
    test_accu_input = np.asarray(test_accu_list, dtype='int32')
    test_law_input = np.asarray(test_law_list, dtype='int32')
    test_term_input = np.asarray(test_term_list, dtype='int32')

    print (test_accu_input.shape)
    print (test_law_input.shape)
    print (test_facts.shape)
    
    
    model.summary()
    model.fit([train_singlefacts, train_accu_input, train_law_input,
               train_term_input,
               train_pair_front_word, train_pair_last_word, train_pair_front_isword, train_pair_last_isword, train_pair_front_num, train_pair_last_num, train_pair_front_isnum, train_pair_last_isnum], {
        'accu_preds': to_categorical(train_accus, num_classes=config.num_accu_liu),
        'law_preds': to_categorical(train_laws, num_classes=config.num_law_liu),
        'term_preds': to_categorical(train_terms, num_classes=config.num_term_liu),
    },
            validation_data=([test_singlefacts, test_accu_input, test_law_input,
                              test_term_input,
                              test_pair_front_word, test_pair_last_word, test_pair_front_isword, test_pair_last_isword, test_pair_front_num, test_pair_last_num, test_pair_front_isnum, test_pair_last_isnum],  {
            'accu_preds': to_categorical(test_accus, num_classes=config.num_accu_liu),
            'law_preds': to_categorical(test_laws, num_classes=config.num_law_liu),
             'term_preds': to_categorical(test_terms, num_classes=config.num_term_liu),
        }),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callback_lists)

