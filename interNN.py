# -*- coding: utf-8 -*-

import gzip
import os
import tempfile

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.contrib import rnn

import math
import random
import time
import datetime

from keras.utils import np_utils

from sklearn import preprocessing

# wide, embedding, co-occurrence and feature interaction 
# version 1: interaction as external hiddens

class wide_embed_coocc_NN():
    
#   build the network graph
    def __init__(self, n_conti, n_disc, n_class,\
                 n_disc_voca, session,n_embedding, n_hidden_list, lr, l2):
        
        
        self.LEARNING_RATE = lr
                
        self.N_CLASS = n_class
        self.N_CONTI = n_conti
        self.N_DISC = n_disc
        self.N_DISC_VOCA = n_disc_voca
        self.N_EMBED = n_embedding
        self.L2 = l2
        
        self.N_HIDDEN_LAYERS = len(n_hidden_list)
        self.N_TOTAL = self.N_CONTI+ self.N_DISC*self.N_EMBED

        self.sess=session
        
        self.dx = tf.placeholder(tf.int32, [None, self.N_DISC])
        self.cx = tf.placeholder(tf.float32, [None, self.N_CONTI])
        self.y = tf.placeholder(tf.float32, [None, self.N_CLASS])
        self.lr = tf.placeholder(tf.float32, shape=())
        self.keep_prob = tf.placeholder(tf.float32)
        
#       wide part on categorical features
        self.dx_trans = tf.transpose( self.dx, [1,0] )
        
#         for i in range(self.N_DISC):
            
#             with tf.variable_scope("wide"+str(i)):
#                 w= tf.Variable(\
#                 tf.random_uniform([self.N_DISC_VOCA[i], self.N_CLASS],-1.0, 1.0) )
# #                         tf.random_normal([self.N_DISC_VOCA, self.N_CLASS],\
# #                         stddev=math.sqrt(2.0/float(self.N_DISC_VOCA))) )

#                 b= tf.Variable(tf.zeros( [ self.N_CLASS ] ))

#                 if i==0:
#                     dx_wsum = tf.nn.embedding_lookup( w, self.dx_trans[i] ) + b
                    
# #                   L2
#                     self.regularizer_wide = tf.nn.l2_loss(w)
                    
#                 else:
#                     dx_wsum = dx_wsum + tf.nn.embedding_lookup( w, self.dx_trans[i] ) + b
                    
# #                   L2  
#                     self.regularizer_wide = self.regularizer_wide + tf.nn.l2_loss(w)
        
        
# #       non-linear
#         dx_wsum = tf.nn.relu( dx_wsum )

        
#       co-occurrence of categorical features
        for i in range(20):
            
            tmpvoca1 = self.N_DISC_VOCA[i] 
            
            for j in range(i+1,20):
                
                tmpvoca2 = self.N_DISC_VOCA[j]
                
                with tf.variable_scope("cooccur"+str(i)+str(j)):
                    
#                   vector approximate of co-occurrence matrix 
                    w1 = tf.Variable(tf.random_normal([self.N_CLASS, tmpvoca1, 1],\
                                            stddev=math.sqrt(2.0/float( tmpvoca1 ))) )
                    
                    w2 = tf.Variable(tf.random_normal([self.N_CLASS, 1, tmpvoca2],\
                                            stddev=math.sqrt(2.0/float( tmpvoca2 ))) )
            
                    
                    b= tf.Variable(tf.zeros( [ self.N_CLASS ] ))

                    
                    cooc_mat = []
                    for k in range(self.N_CLASS):
                        cooc_mat.append( tf.reshape( tf.matmul(w1[k],w2[k]), [-1,1] ) )
                        
#                   n_class by voca1*voca2  
                    cooc_vec = tf.stack(cooc_mat)
                    cooc_vec = tf.reshape(cooc_vec, [self.N_CLASS, -1] )
                    cooc_vec = tf.transpose(cooc_vec,[1,0])
                    
#                   build the idx of co-occurrence features
                    cooc_idx = self.dx_trans[i] * tmpvoca2 + self.dx_trans[j]
    
                    if i==0 and j==1:
                        cooc_wsum  = tf.nn.embedding_lookup( cooc_vec, cooc_idx ) + b
#                       L2
                        self.regularizer_cooc   = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
                    else:
                        cooc_wsum += tf.nn.embedding_lookup( cooc_vec, cooc_idx ) + b
#                       L2
                        self.regularizer_cooc += ( tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) )
    
        
#       non-linear
        cooc_wsum = tf.nn.relu( cooc_wsum )

    
#       feature interaction
        for i in range(self.N_DISC):
            
            with tf.variable_scope("interact"+str(i)):
                w= tf.Variable(\
                        tf.random_normal([self.N_DISC_VOCA[i], self.N_CONTI],\
                        stddev=math.sqrt(2.0/float(self.N_DISC_VOCA[i]))) )
#                     tf.random_uniform([self.N_DISC_VOCA, self.N_EMBED],-1.0, 1.0 )
        
                if i==0:
                    inter_wsum  = tf.nn.embedding_lookup( w, self.dx_trans[i] )
#                   L2 
                    self.regularizer_inter  = tf.nn.l2_loss(w)
            
                else:
                    inter_wsum += tf.nn.embedding_lookup( w, self.dx_trans[i] )
#                   L2 
                    self.regularizer_inter += tf.nn.l2_loss(w)

    
#       add noise bias
        singleCol_h = tf.reduce_sum( self.cx * inter_wsum, 1 )
        singleCol_h = tf.reshape( singleCol_h, [-1,1] )
        
#       interaction hidden layers
        with tf.variable_scope("inter_h0"):
            w = tf.Variable(tf.random_normal([1, n_hidden_list[0]],\
                        stddev=math.sqrt(2.0/float(n_hidden_list[0])))) 
                        
            b = tf.Variable(tf.zeros( [ n_hidden_list[0] ] ))
            
            h_inter = tf.matmul( singleCol_h, w )
            h_inter = tf.nn.relu( h_inter + b)
#           L2  
            self.regularizer_inter += tf.nn.l2_loss(w)
            
        for i in range(1, self.N_HIDDEN_LAYERS):
            
            with tf.variable_scope("inter_h"+str(i)):
                w = tf.Variable(\
                        tf.random_normal([n_hidden_list[i-1],n_hidden_list[i]],\
                                stddev=math.sqrt(2.0/float(n_hidden_list[i-1])))) 
                b = tf.Variable(tf.zeros( [n_hidden_list[i]] ))
                h_inter = tf.nn.relu( tf.add( tf.matmul(h_inter, w),b) )
#               L2
                self.regularizer_inter += tf.nn.l2_loss(w)
    
    
#       embedding of categorical features        
        tmp_embeded = []                
        for i in range(self.N_DISC):
            
            with tf.variable_scope("embedding"+str(i)):
                w= tf.Variable(\
                        tf.random_normal([self.N_DISC_VOCA[i], self.N_EMBED],\
                        stddev=math.sqrt(2.0/float(self.N_DISC_VOCA[i]))) )
#                     tf.random_uniform([self.N_DISC_VOCA, self.N_EMBED],-1.0, 1.0 )
                              
                tmp_embeded.append( tf.nn.embedding_lookup( w, self.dx_trans[i] ) )
                
        dx_embeded = tf.concat(tmp_embeded, 1)
        x_concate =  tf.concat( [self.cx,dx_embeded], 1 )
        
#       embedding + continuous hidden layers
        with tf.variable_scope("h0"):
            w= tf.Variable(tf.random_normal([self.N_TOTAL, n_hidden_list[0]],\
                        stddev=math.sqrt(2.0/float(self.N_CONTI)))) 
            b= tf.Variable(tf.zeros( [ n_hidden_list[0] ] ))
            h = tf.nn.relu( tf.add( tf.matmul(x_concate, w),b) )
#           L2
            self.regularizer = tf.nn.l2_loss(w)
        
                
        for i in range(1, self.N_HIDDEN_LAYERS):
            
            with tf.variable_scope("layer"+str(i)):
                w= tf.Variable(\
                        tf.random_normal([n_hidden_list[i-1],n_hidden_list[i]],\
                                stddev=math.sqrt(2.0/float(n_hidden_list[i-1])))) 
                b= tf.Variable(tf.zeros( [n_hidden_list[i]] ))
                h = tf.nn.relu( tf.add( tf.matmul(h, w),b) )
                
#               L2
                self.regularizer += tf.nn.l2_loss(w)
        
                
#       dropout
#       h = tf.nn.dropout(h, self.keep_prob)

        
#       output layer  
        h = tf.concat([h, h_inter], 1)
    
        with tf.variable_scope("temp"):
            w= tf.Variable(tf.random_normal([n_hidden_list[self.N_HIDDEN_LAYERS-1]*2,\
                                             n_hidden_list[self.N_HIDDEN_LAYERS-1] ],\
                        stddev=math.sqrt(2.0/float(n_hidden_list[self.N_HIDDEN_LAYERS-1]))))
                           
            b= tf.Variable(tf.zeros( [ n_hidden_list[self.N_HIDDEN_LAYERS-1] ] ))
            h = tf.nn.relu( tf.add( tf.matmul(h, w),b) )
                           
#           L2  
            self.regularizer += tf.nn.l2_loss(w) 

    
        with tf.variable_scope("output"):
            
            w= tf.Variable(tf.random_normal([n_hidden_list[self.N_HIDDEN_LAYERS-1],\
                                             self.N_CLASS],\
            stddev=math.sqrt(2.0/float(n_hidden_list[self.N_HIDDEN_LAYERS-1])))) 
            b= tf.Variable(tf.zeros( [ self.N_CLASS ] ))
            
            h = tf.matmul(h, w)
#             h = tf.add( h, dx_wsum ) #wide
            h = tf.add( h, cooc_wsum ) #co-occurrence
            h = tf.add( h, b )

            self.logit = h
     
    
#           L2  
            self.regularizer += tf.nn.l2_loss(w) 
    
#       overall L2
        self.regularizer += (self.regularizer_cooc + self.regularizer_inter)
#     self.regularizer_wide + 
         
    
#   initialize loss and optimization operations for training
    def train_ini(self):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                                    logits=self.logit, labels=self.y)) \
                                  + self.L2*self.regularizer
        self.optimizer = \
        tf.train.AdadeltaOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#       tf.train.RMSPropOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.AdadeltaOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.cost)
        
        self.init = tf.global_variables_initializer()
        sess.run( self.init )
        
#   training on batch of data
    def train_batch(self, dx_batch,cx_batch, y_batch, keep_prob, lr):
        
        _,c = sess.run([self.optimizer,self.cost],\
                       feed_dict={self.dx:dx_batch, \
                                  self.cx:cx_batch, self.y:y_batch,\
                                  self.lr:lr,\
                                  self.keep_prob:keep_prob\
                                 })
        
        return c
    
#   initialize inference         
    def inference_ini(self):
        self.correct_prediction = tf.equal(tf.argmax(self.logit,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        
#   infer givn testing data    
    def inference(self, dx_test, cx_test, y_test, keep_prob):
        return sess.run([self.accuracy], feed_dict={self.dx:dx_test,\
                                                  self.cx:cx_test, self.y:y_test,\
                                                  self.keep_prob:keep_prob\
                                                   })
        
#   unit_test
    def test(self, dx_test, cx_test, y_test ):
        self.init = tf.global_variables_initializer()
        sess.run( self.init )
        return sess.run( [self.tmpshape], \
                 feed_dict={self.dx:dx_test, self.cx:cx_test, self.y:y_test})



# wide, embedding, co-occurrence and feature interaction 
# version 2: individual hidden layers on embedding and interaction

class InterNN_IndiH():
    

    
    
    def __init__(self, n_conti, n_disc, n_class,\
                 n_disc_voca, session,n_embedding, n_hidden_list, lr, l2, batch_size):
        
#       build the network graph 
        self.LEARNING_RATE = lr
                
        self.N_CLASS = n_class
        self.N_CONTI = n_conti
        self.N_DISC = n_disc
        self.N_DISC_VOCA = n_disc_voca
        self.N_EMBED = n_embedding
        self.N_BATCH = batch_size
        self.L2 = l2
        
        self.N_HIDDEN_LAYERS = len(n_hidden_list)
        self.N_TOTAL = self.N_CONTI+ self.N_DISC*self.N_EMBED
        self.N_TOTAL_EMBED = self.N_DISC*self.N_EMBED

        self.sess=session
        
        self.dx = tf.placeholder(tf.int32, [None, self.N_DISC])
        self.cx = tf.placeholder(tf.float32, [None, self.N_CONTI])
        self.y = tf.placeholder(tf.float32, [None, self.N_CLASS])
        self.lr = tf.placeholder(tf.float32, shape=())
        self.keep_prob = tf.placeholder(tf.float32)
        
#       wide part on categorical features
        self.dx_trans = tf.transpose( self.dx, [1,0] )
    
    
    
#         for i in range(self.N_DISC):
            
#             with tf.variable_scope("wide"+str(i)):
#                 w= tf.Variable(\
#                 tf.random_uniform([self.N_DISC_VOCA[i], self.N_CLASS],-1.0, 1.0) )
# #                         tf.random_normal([self.N_DISC_VOCA, self.N_CLASS],\
# #                         stddev=math.sqrt(2.0/float(self.N_DISC_VOCA))) )
#                 if i==0:
#                     dx_wsum = tf.nn.embedding_lookup( w, self.dx_trans[i] )
                    
# #                   L2
#                     self.regularizer_wide = tf.nn.l2_loss(w)
                    
#                 else:
#                     dx_wsum = dx_wsum + tf.nn.embedding_lookup( w, self.dx_trans[i] )
                    
# #                   L2  
#                     self.regularizer_wide = self.regularizer_wide + tf.nn.l2_loss(w)
               
    
    
    
#       co-occurrence of categorical features
        for i in range(20):
            
            tmpvoca1 = self.N_DISC_VOCA[i] 
            
            for j in range(i+1,20):
                
                tmpvoca2 = self.N_DISC_VOCA[j]
                
                with tf.variable_scope("cooccur"+str(i)+str(j)):
                    
                    
#                   vector approximate of co-occurrence matrix 
                    w1 = tf.Variable(tf.random_normal([self.N_CLASS, tmpvoca1, 1],\
                                            stddev=math.sqrt(2.0/float( tmpvoca1 ))) )
                    
                    w2 = tf.Variable(tf.random_normal([self.N_CLASS, 1, tmpvoca2],\
                                            stddev=math.sqrt(2.0/float( tmpvoca2 ))) )
            
                    
                    b= tf.Variable(tf.zeros( [ self.N_CLASS ] ))

                    
                    cooc_mat = []
                    for k in range(self.N_CLASS):
                        cooc_mat.append( tf.reshape( tf.matmul(w1[k],w2[k]), [-1,1] ) )
                        
#                   n_class by voca1*voca2  
        
                    cooc_vec = tf.stack(cooc_mat)
                    cooc_vec = tf.reshape(cooc_vec, [self.N_CLASS, -1] )
                    cooc_vec = tf.transpose(cooc_vec,[1,0])
                    
#                   build the idx of co-occurrence features
                    cooc_idx = self.dx_trans[i] * tmpvoca2 + self.dx_trans[j]
    
                    if i==0 and j==1:
                        cooc_wsum  = tf.nn.embedding_lookup( cooc_vec, cooc_idx ) + b
#                       L2 
                        self.regularizer_cooc   = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) 
                    else:
                        cooc_wsum += tf.nn.embedding_lookup( cooc_vec, cooc_idx ) + b
#                       L2
                        self.regularizer_cooc += (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2))
    
#       non-linear 
        cooc_wsum = tf.nn.relu( cooc_wsum )
        
        
        
#       embedding of categorical features        
        tmp_embeded = []                
        for i in range(self.N_DISC):
            
            with tf.variable_scope("embedding"+str(i)):
                w= tf.Variable(\
                        tf.random_normal([self.N_DISC_VOCA[i], self.N_EMBED],\
                        stddev=math.sqrt(2.0/float(self.N_DISC_VOCA[i]))) )
#                     tf.random_uniform([self.N_DISC_VOCA, self.N_EMBED],-1.0, 1.0 )
                              
                tmp_embeded.append( tf.nn.embedding_lookup( w, self.dx_trans[i] ) )
                
        dx_embeded = tf.concat(tmp_embeded, 1)
        
        
#       embedding hidden layers
        with tf.variable_scope("embed_h0"):
            w= tf.Variable(tf.random_normal([self.N_TOTAL_EMBED, n_hidden_list[0]],\
                        stddev=math.sqrt(2.0/float(self.N_CONTI)))) 
            b= tf.Variable(tf.zeros( [ n_hidden_list[0] ] ))
            h_embed = tf.nn.relu( tf.add( tf.matmul(dx_embeded, w),b) )
            
#           L2   
            self.regularizer = tf.nn.l2_loss(w)
                
        for i in range(1, self.N_HIDDEN_LAYERS):
            
            with tf.variable_scope("embed_h"+str(i)):
                w= tf.Variable(\
                        tf.random_normal([n_hidden_list[i-1],n_hidden_list[i]],\
                                stddev=math.sqrt(2.0/float(n_hidden_list[i-1])))) 
                b= tf.Variable(tf.zeros( [n_hidden_list[i]] ))
                h_embed = tf.nn.relu( tf.add( tf.matmul(h_embed, w),b) )       
            
#               L2   
                self.regularizer += tf.nn.l2_loss(w)
        
        
        
        
        
#       feature interaction
        for i in range(self.N_DISC):
            
            with tf.variable_scope("interact"+str(i)):
                w= tf.Variable(\
                        tf.random_normal([self.N_DISC_VOCA[i], self.N_CONTI],\
                        stddev=math.sqrt(2.0/float(self.N_DISC_VOCA[i]))) )
#                     tf.random_uniform([self.N_DISC_VOCA, self.N_EMBED],-1.0, 1.0 )
        
                if i==0:
                    inter_wsum  = tf.nn.embedding_lookup( w, self.dx_trans[i] )
#                   L2 
                    self.regularizer_inter  = tf.nn.l2_loss(w)
            
                else:
                    inter_wsum += tf.nn.embedding_lookup( w, self.dx_trans[i] )
#                   L2 
                    self.regularizer_inter += tf.nn.l2_loss(w)

#       add noise bias
        singleCol_h = tf.reduce_sum( self.cx * inter_wsum, 1   )
        tmp_h=[]
        for i in range( n_hidden_list[0] ):
            with tf.variable_scope("noise"+str( i )):
                noise_b = tf.Variable(tf.random_normal([ ],\
                        stddev=math.sqrt(2.0/float(n_hidden_list[0]))))
                tmp_h.append( singleCol_h + noise_b ) 
                
#               L2  
                self.regularizer_inter += tf.nn.l2_loss(noise_b)
        
#         tmp_h=[]
#         for i in range( n_hidden_list[0] ):
#             with tf.variable_scope("noise"+str( i )):
#                 noise_b = tf.Variable(tf.random_normal([ self.N_CONTI ],\
#                         stddev=math.sqrt(2.0/float(n_hidden_list[0]))))
#                 tmp_h.append( tf.reduce_sum( self.cx * (inter_wsum + noise_b), 1 ) ) 
        
        tmp_h = tf.stack( tmp_h )
        tmp_h = tf.transpose( tmp_h, [1,0])
            
        
#       interaction hidden layers


#         singleCol_h = tf.reduce_sum( self.cx * inter_wsum, 1   )
        
#         tmp_h = [ singleCol_h ]*n_hidden_list[0]
        
#         tmp_h = tf.stack( tmp_h )
#         tmp_h = tf.transpose( tmp_h, [1,0])
        
        
        with tf.variable_scope("inter_h0"):
            w = tf.Variable(tf.random_normal([self.N_CONTI, n_hidden_list[0]],\
                        stddev=math.sqrt(2.0/float(self.N_CONTI)))) 
                        
            b = tf.Variable(tf.zeros( [ n_hidden_list[0] ] ))
            
            h_inter = tf.matmul(self.cx, w)
            h_inter = h_inter + tmp_h
            h_inter = tf.nn.relu( h_inter + b)
                             
#           L2  
            self.regularizer += tf.nn.l2_loss(w)
            
        for i in range(1, self.N_HIDDEN_LAYERS):
            
            with tf.variable_scope("inter_h"+str(i)):
                w = tf.Variable(\
                        tf.random_normal([n_hidden_list[i-1],n_hidden_list[i]],\
                                stddev=math.sqrt(2.0/float(n_hidden_list[i-1])))) 
                b = tf.Variable(tf.zeros( [n_hidden_list[i]] ))
                h_inter = tf.nn.relu( tf.add( tf.matmul(h_inter, w),b) )
        
#               L2
                self.regularizer += tf.nn.l2_loss(w)
    
    
#       dropout
#       h = tf.nn.dropout(h, self.keep_prob)
        
        
#       output layer  
        h = tf.concat([h_embed, h_inter], 1)
        
        with tf.variable_scope("output"):
            
            w = tf.Variable(tf.random_normal([n_hidden_list[self.N_HIDDEN_LAYERS-1]*2,\
                                             self.N_CLASS],\
            stddev=math.sqrt(2.0/float(n_hidden_list[self.N_HIDDEN_LAYERS-1])))) 
            b = tf.Variable(tf.zeros( [ self.N_CLASS ] ))
            
            h = tf.matmul(h, w)
#             h = tf.add( h, dx_wsum )
            h = tf.add( h, cooc_wsum )
            h = tf.add( h, b )
        
            self.logit = h
            
#           L2  
            self.regularizer += tf.nn.l2_loss(w) 
    
    
#       overall L2
        self.regularizer += (self.regularizer_cooc + self.regularizer_inter)
#       self.regularizer_wide + 
         
        
#   initialize loss and optimization operations for training
    def train_ini(self):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                                    logits=self.logit, labels=self.y) ) \
                                  + self.L2*self.regularizer
        self.optimizer = \
        tf.train.AdadeltaOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#       tf.train.RMSPropOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.AdadeltaOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.cost)
        
        self.init = tf.global_variables_initializer()
        sess.run( self.init )
        
#   training on batch of data
    def train_batch(self, dx_batch,cx_batch, y_batch, keep_prob, lr):
        
        _,c = sess.run([self.optimizer,self.cost],\
                       feed_dict={self.dx:dx_batch, \
                                  self.cx:cx_batch, self.y:y_batch,\
                                  self.lr:lr,\
                                  self.keep_prob:keep_prob\
                                 })
        
        return c
        
#   initialize inference         
    def inference_ini(self):
        self.correct_prediction = tf.equal(tf.argmax(self.logit,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        
#   infer givn testing data    
    def inference(self, dx_test, cx_test, y_test, keep_prob):
        return sess.run([self.accuracy], feed_dict={self.dx:dx_test,\
                                                  self.cx:cx_test, self.y:y_test,\
                                                  self.keep_prob:keep_prob\
                                                   })
        
#   unit_test
    def test(self, dx_test, cx_test, y_test ):
        self.init = tf.global_variables_initializer()
        sess.run( self.init )
        return sess.run( [self.tmp1], \
                 feed_dict={self.dx:dx_test, self.cx:cx_test, self.y:y_test})
        

# wide, embedding, co-occurrence and feature interaction
# version 3: fuse all into hidden layers

class InterNN_fuse():
    
    
    def __init__(self, n_conti, n_disc, n_class,\
                 n_disc_voca, session,n_embedding, n_hidden_list, lr, l2, batch_size, \
                 max_norm ):
        
#       build the network graph 
        self.LEARNING_RATE = lr
                
        self.N_CLASS = n_class
        self.N_CONTI = n_conti
        self.N_DISC  = n_disc
        self.N_DISC_VOCA = n_disc_voca
        self.N_EMBED = n_embedding
        self.N_BATCH = batch_size
        self.L2 = l2
        
        self.N_HIDDEN_LAYERS = len(n_hidden_list)
        self.N_TOTAL = self.N_CONTI+ self.N_DISC*self.N_EMBED
        self.N_TOTAL_EMBED = self.N_DISC*self.N_EMBED
   
        self.MAX_NORM = max_norm
        self.epsilon = 1e-3
        
        self.sess = session
        
        self.dx = tf.placeholder(tf.int32, [None, self.N_DISC])
        self.cx = tf.placeholder(tf.float32, [None, self.N_CONTI])
        self.y  = tf.placeholder(tf.float32, [None, self.N_CLASS])
        self.lr = tf.placeholder(tf.float32, shape=())
        self.keep_prob = tf.placeholder(tf.float32)
        
#       wide part on categorical features
        self.dx_trans = tf.transpose( self.dx, [1,0] )
    
        for i in range(self.N_DISC):
            
            with tf.variable_scope("wide"+str(i)):
#               !change
                w= tf.Variable(\
                               tf.random_normal([self.N_DISC_VOCA[i], self.N_CLASS],\
                        stddev = math.sqrt(2.0/float(self.N_DISC_VOCA[i]))) )
#               tf.random_uniform([self.N_DISC_VOCA[i], self.N_CLASS],-1.0, 1.0) )

#               max norm constraint
#                 w = tf.clip_by_norm( w, self.MAX_NORM, axes = 1)

#               !! change
                b  = tf.Variable(tf.zeros( [ self.N_CLASS ] ))

                if i==0:
                    dx_wsum = tf.nn.embedding_lookup( w, self.dx_trans[i] ) + b
#                   L2
                    self.regularizer_wide  =  tf.nn.l2_loss(w)
                    
                else:
                    dx_wsum = dx_wsum + tf.nn.embedding_lookup( w, self.dx_trans[i] ) + b
#                   L2  
                    self.regularizer_wide += tf.nn.l2_loss(w)
        
    
#       nonlinear activation
#         dx_wsum = tf.nn.relu( dx_wsum )
        dx_wsum = tf.maximum( 0.01*dx_wsum, dx_wsum )
               
    
    
#       co-occurrence of categorical features
        for i in range(20):
            
            tmpvoca1 = self.N_DISC_VOCA[i] 
            
            for j in range(i+1,20):
                
                tmpvoca2 = self.N_DISC_VOCA[j]
                
                with tf.variable_scope("cooccur"+str(i)+str(j)):
                    
#                   vector approximate of co-occurrence matrix 
                    w1 = tf.Variable(tf.random_normal([self.N_CLASS, tmpvoca1, 1],\
                                            stddev=math.sqrt(2.0/float( tmpvoca1 ))) )
                    
                    w2 = tf.Variable(tf.random_normal([self.N_CLASS, 1, tmpvoca2],\
                                            stddev=math.sqrt(2.0/float( tmpvoca2 ))) )
                    
#                   !!! change
                    b  = tf.Variable(tf.zeros( [ self.N_CLASS ] ))
                    
                    cooc_mat = []
                    for k in range(self.N_CLASS):
                        cooc_mat.append( tf.reshape( tf.matmul(w1[k],w2[k]), [-1,1] ) )
                        
#                   n_class by voca1*voca2  
                    cooc_vec = tf.stack(cooc_mat)
                    cooc_vec = tf.reshape(cooc_vec, [self.N_CLASS, -1] )
                    cooc_vec = tf.transpose(cooc_vec,[1,0])
            
#                   max norm constraint
#                     cooc_vec = tf.clip_by_norm( cooc_vec, self.MAX_NORM, axes = 0)
    
                    
#                   build the idx of co-occurrence features
                    cooc_idx = self.dx_trans[i] * tmpvoca2 + self.dx_trans[j]
    
                    if i==0 and j==1:
#                       co-occurrence weighted sum

#                       !!! change
                        cooc_wsum  = tf.nn.embedding_lookup( cooc_vec, cooc_idx ) + b
#                       L2 
                        self.regularizer_cooc  = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) 
                    else:
            
#                       !!! change
                        cooc_wsum += (tf.nn.embedding_lookup( cooc_vec, cooc_idx ) + b) 
#                       L2
                        self.regularizer_cooc += (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2))
    
#       nonlinear activation
#         cooc_wsum = tf.nn.relu( cooc_wsum )
        cooc_wsum = tf.maximum( 0.01*cooc_wsum, cooc_wsum )
        
        
#       embedding of categorical features        
        tmp_embeded = []                
        for i in range(self.N_DISC):
            
            with tf.variable_scope("embedding"+str(i)):
                w= tf.Variable(\
                        tf.random_normal([self.N_DISC_VOCA[i], self.N_EMBED],\
                        stddev = math.sqrt(2.0/float(self.N_DISC_VOCA[i]))) )
#                     tf.random_uniform([self.N_DISC_VOCA, self.N_EMBED],-1.0, 1.0 )
                              
                tmp_embeded.append( tf.nn.embedding_lookup( w, self.dx_trans[i] ) )
                
        dx_embeded = tf.concat(tmp_embeded, 1)
        x_concate  = tf.concat( [self.cx,dx_embeded], 1 )
        
        
#       feature interaction
        for i in range(self.N_DISC):
            
            with tf.variable_scope("interact"+str(i)):
                w= tf.Variable(\
                        tf.random_normal([self.N_DISC_VOCA[i], self.N_CONTI],\
                        stddev=math.sqrt(2.0/float(self.N_DISC_VOCA[i]))) )
#                     tf.random_uniform([self.N_DISC_VOCA, self.N_EMBED],-1.0, 1.0 )
        
#               max norm constraint
#                 w = tf.clip_by_norm( w, self.MAX_NORM, axes = 1)
            
            
                if i==0:
                    inter_wsum  = tf.nn.embedding_lookup( w, self.dx_trans[i] )
#                   L2 
                    self.regularizer_inter  = tf.nn.l2_loss(w)
            
                else:
                    inter_wsum += tf.nn.embedding_lookup( w, self.dx_trans[i] )
#                   L2 
                    self.regularizer_inter += tf.nn.l2_loss(w)

    
#       add noise bias
        singleCol_h = tf.reduce_sum( self.cx * inter_wsum, 1   )
        tmp_h=[]
        for i in range( n_hidden_list[0] ):
            with tf.variable_scope("noise"+str( i )):
                noise_b = tf.Variable(tf.random_normal([ ],\
                        stddev=math.sqrt(2.0/float(n_hidden_list[0]))))
                
                tmp_h.append( singleCol_h + noise_b ) 
        
#         tmp_h=[]
#         for i in range( n_hidden_list[0] ):
#             with tf.variable_scope("noise"+str( i )):
#                 noise_b = tf.Variable(tf.random_normal([ self.N_CONTI ],\
#                         stddev=math.sqrt(2.0/float(n_hidden_list[0]))))
#                 tmp_h.append( tf.reduce_sum( self.cx * (inter_wsum + noise_b), 1 ) ) 
        
        
        tmp_h = tf.stack( tmp_h )
        tmp_h = tf.transpose( tmp_h, [1,0])
        
        
#       dropout
#         h_inter = tf.nn.dropout(h_inter, self.keep_prob)
        
        
#       hidden layers on fusion
        with tf.variable_scope("inter_h0"):
            
#          conventional one   
            w= tf.Variable(tf.random_normal([self.N_TOTAL, n_hidden_list[0]],\
                        stddev=math.sqrt(2.0/float(self.N_TOTAL)))) 
            b= tf.Variable(tf.zeros( [ n_hidden_list[0] ] ))
        
#           max norm constraint
#             w = tf.clip_by_norm( w, self.MAX_NORM, axes = 0) 
            
            h_inter = tf.matmul(x_concate, w)
            h_inter = h_inter + tmp_h
#           h_inter = tf.add( h_inter, b )
            
            
#           batch normalization 
            batch_m, batch_v = tf.nn.moments(h_inter,[0])
            scale = tf.Variable( tf.ones( [ n_hidden_list[0] ]) )
            beta  = tf.Variable( tf.zeros([ n_hidden_list[0] ]) )
            
            h_inter = \
            tf.nn.batch_normalization(h_inter, batch_m, batch_v, beta, scale,\
                                      self.epsilon)
            
#           nonlinear activation   
#             h_inter = tf.nn.relu( h_inter )
            h_inter = tf.maximum( 0.01*h_inter, h_inter )
            
#           L2   
            self.regularizer = tf.nn.l2_loss(w)
        
#       dropout
#         h_inter = tf.nn.dropout(h_inter, self.keep_prob)
        
        
        
        for i in range(1, self.N_HIDDEN_LAYERS):
            
            with tf.variable_scope("inter_h"+str(i)):
                w = tf.Variable(\
                        tf.random_normal([n_hidden_list[i-1],n_hidden_list[i]],\
                                stddev=math.sqrt(2.0/float(n_hidden_list[i-1])))) 
                b = tf.Variable(tf.zeros( [n_hidden_list[i]] ))
                
#               max norm constraint
#                 w = tf.clip_by_norm( w, self.MAX_NORM,axes = 0)
        
                h_inter = tf.matmul(h_inter, w)
#               h_inter = tf.add(h_inter,b)
                
                
#               batch normalization 
                batch_m, batch_v = tf.nn.moments(h_inter,[0])
                scale = tf.Variable( tf.ones( [ n_hidden_list[i] ]) )
                beta  = tf.Variable( tf.zeros([ n_hidden_list[i] ]) )
            
                h_inter = \
                tf.nn.batch_normalization(h_inter, batch_m, batch_v, beta, scale,\
                                          self.epsilon)

#               nonlinear activation  
                h_inter = tf.maximum( 0.01*h_inter, h_inter )
#                 h_inter = tf.nn.relu( h_inter )
        
#               L2
                self.regularizer += tf.nn.l2_loss(w)
    
#               dropout
#                 h_inter = tf.nn.dropout(h_inter, self.keep_prob)
        
        
#       output layer  
        h = h_inter
        
        with tf.variable_scope("output"):
            
            w = tf.Variable(tf.random_normal([n_hidden_list[self.N_HIDDEN_LAYERS-1],\
                                             self.N_CLASS],\
            stddev = math.sqrt(2.0/float(n_hidden_list[self.N_HIDDEN_LAYERS-1])))) 
            b = tf.Variable(tf.zeros( [ self.N_CLASS ] ))
            
            h = tf.matmul(h, w)
            
            h = tf.add( h, dx_wsum )
            h = tf.add( h, cooc_wsum )
            h = tf.add( h, b )
        
            self.logit = h
            
#           L2  
            self.regularizer += tf.nn.l2_loss(w) 
    
#       overall L2
        self.regularizer += \
        (self.regularizer_cooc + self.regularizer_inter + self.regularizer_wide)

        
#   initialize loss and optimization operations for training
    def train_ini(self):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                                    logits=self.logit, labels=self.y)) \
                                   + self.L2*self.regularizer
            
        self.optimizer = \
        tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.AdadeltaOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.RMSPropOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.AdadeltaOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.cost)
        
        self.init = tf.global_variables_initializer()
        sess.run( self.init )
        
#   training on batch of data
    def train_batch(self, dx_batch,cx_batch, y_batch, keep_prob, lr):
        
        _,c = sess.run([self.optimizer,self.cost],\
                        feed_dict={self.dx:dx_batch,\
                                   self.cx:cx_batch, self.y:y_batch,\
                                   self.lr:lr,\
                                   self.keep_prob:keep_prob\
                                 })
        
        return c
        
#   initialize inference         
    def inference_ini(self):
        self.correct_prediction = tf.equal(tf.argmax(self.logit,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        
#   infer givn testing data    
    def inference(self, dx_test, cx_test, y_test, keep_prob):
        return sess.run([self.accuracy], feed_dict={self.dx:dx_test,\
                                                    self.cx:cx_test, self.y:y_test,\
                                                    self.keep_prob:keep_prob\
                                                   })
        
#   unit_test
    def test(self, dx_test, cx_test, y_test ):
        self.init = tf.global_variables_initializer()
        sess.run( self.init )
        return sess.run( [self.tmp1], \
                 feed_dict={self.dx:dx_test, self.cx:cx_test, self.y:y_test})
    
    
    
    
# wide, embedding, co-occurrence and feature interaction
# version 4: fuse all into hidden layers, with orthogonal initialization

class InterNN_fuse_ortho():
    
    
#   orthogonal initialization of weights 
#   return row vector  
    def orthogonal(self, shape):

        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        
        return q.reshape(shape)
    
    
    def __init__(self, n_conti, n_disc, n_class,\
                 n_disc_voca, session,n_embedding, n_hidden_list, lr, l2, batch_size, \
                 max_norm ):
        
#       build the network graph 
        self.LEARNING_RATE = lr
                
        self.N_CLASS = n_class
        self.N_CONTI = n_conti
        self.N_DISC  = n_disc
        self.N_DISC_VOCA = n_disc_voca
        self.N_EMBED = n_embedding
        self.N_BATCH = batch_size
        self.L2 = l2
        
        self.N_HIDDEN_LAYERS = len(n_hidden_list)
        self.N_TOTAL = self.N_CONTI+ self.N_DISC*self.N_EMBED
        self.N_TOTAL_EMBED = self.N_DISC*self.N_EMBED
   
        self.MAX_NORM = max_norm
        self.epsilon = 1e-3
        
        self.sess = session
        
        self.dx = tf.placeholder(tf.int32, [None, self.N_DISC])
        self.cx = tf.placeholder(tf.float32, [None, self.N_CONTI])
        self.y  = tf.placeholder(tf.float32, [None, self.N_CLASS])
        self.lr = tf.placeholder(tf.float32, shape=())
        self.keep_prob = tf.placeholder(tf.float32)
        
#       wide part on categorical features
        self.dx_trans = tf.transpose( self.dx, [1,0] )
    
        for i in range(self.N_DISC):
            
            with tf.variable_scope("wide"+str(i)):
#               !change
                w= tf.Variable(\
                               tf.random_normal([self.N_DISC_VOCA[i], self.N_CLASS],\
                        stddev = math.sqrt(2.0/float(self.N_DISC_VOCA[i]))) )
#               tf.constant(orthogonal( [self.N_DISC_VOCA[i], self.N_CLASS] )) )
#               tf.random_uniform([self.N_DISC_VOCA[i], self.N_CLASS],-1.0, 1.0) )

#               max norm constraint
#                 w = tf.clip_by_norm( w, self.MAX_NORM, axes = 1)

                if i==0:
                    dx_wsum = tf.nn.embedding_lookup( w, self.dx_trans[i] )
#                   L2
                    self.regularizer_wide  =  tf.nn.l2_loss(w)
                    
                else:
                    dx_wsum = dx_wsum + tf.nn.embedding_lookup( w, self.dx_trans[i] )
#                   L2  
                    self.regularizer_wide += tf.nn.l2_loss(w)
        
    
#       nonlinear activation
#         dx_wsum = tf.nn.relu( dx_wsum )
        dx_wsum = tf.maximum( 0.01*dx_wsum, dx_wsum )
               
    
    
#       co-occurrence of categorical features
        for i in range(20):
            
            tmpvoca1 = self.N_DISC_VOCA[i] 
            
            for j in range(i+1,20):
                
                tmpvoca2 = self.N_DISC_VOCA[j]
                
                with tf.variable_scope("cooccur"+str(i)+str(j)):
                    
#                   vector approximate of co-occurrence matrix 
                    w1 = tf.Variable(tf.random_normal([self.N_CLASS, tmpvoca1, 1],\
                                            stddev=math.sqrt(2.0/float( tmpvoca1 ))) )
                    
                    w2 = tf.Variable(tf.random_normal([self.N_CLASS, 1, tmpvoca2],\
                                            stddev=math.sqrt(2.0/float( tmpvoca2 ))) )
                    
#                   !!! change
                    b  = tf.Variable(tf.zeros( [ self.N_CLASS ] ))
                    
                    cooc_mat = []
                    for k in range(self.N_CLASS):
                        cooc_mat.append( tf.reshape( tf.matmul(w1[k],w2[k]), [-1,1] ) )
                        
#                   n_class by voca1*voca2  
                    cooc_vec = tf.stack(cooc_mat)
                    cooc_vec = tf.reshape(cooc_vec, [self.N_CLASS, -1] )
                    cooc_vec = tf.transpose(cooc_vec,[1,0])
            
#                   max norm constraint
#                     cooc_vec = tf.clip_by_norm( cooc_vec, self.MAX_NORM, axes = 0)
    
                    
#                   build the idx of co-occurrence features
                    cooc_idx = self.dx_trans[i] * tmpvoca2 + self.dx_trans[j]
    
                    if i==0 and j==1:
#                       co-occurrence weighted sum

#                       !!! change
                        cooc_wsum  = tf.nn.embedding_lookup( cooc_vec, cooc_idx )
#                       L2 
                        self.regularizer_cooc  = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) 
                    else:
            
#                       !!! change
                        cooc_wsum += tf.nn.embedding_lookup( cooc_vec, cooc_idx )
#                       L2
                        self.regularizer_cooc += (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2))
    
#       nonlinear activation
#         cooc_wsum = tf.nn.relu( cooc_wsum )
        cooc_wsum = tf.maximum( 0.01*cooc_wsum, cooc_wsum )
        
        
#       embedding of categorical features        
        tmp_embeded = []                
        for i in range(self.N_DISC):
            
            with tf.variable_scope("embedding"+str(i)):
                w= tf.Variable(\
                        tf.random_normal([self.N_DISC_VOCA[i], self.N_EMBED],\
                        stddev = math.sqrt(2.0/float(self.N_DISC_VOCA[i]))) )
#                     tf.random_uniform([self.N_DISC_VOCA, self.N_EMBED],-1.0, 1.0 )
                              
                tmp_embeded.append( tf.nn.embedding_lookup( w, self.dx_trans[i] ) )
                
        dx_embeded = tf.concat(tmp_embeded, 1)
        x_concate  = tf.concat( [self.cx,dx_embeded], 1 )
        
        
#       feature interaction
        for i in range(self.N_DISC):
            
            with tf.variable_scope("interact"+str(i)):
                w= tf.Variable(\
                        tf.random_normal([self.N_DISC_VOCA[i], self.N_CONTI],\
                        stddev=math.sqrt(2.0/float(self.N_DISC_VOCA[i]))) )
#                     tf.random_uniform([self.N_DISC_VOCA, self.N_EMBED],-1.0, 1.0 )
        
#               max norm constraint
#                 w = tf.clip_by_norm( w, self.MAX_NORM, axes = 1)
            
            
                if i==0:
                    inter_wsum  = tf.nn.embedding_lookup( w, self.dx_trans[i] )
#                   L2 
                    self.regularizer_inter  = tf.nn.l2_loss(w)
            
                else:
                    inter_wsum += tf.nn.embedding_lookup( w, self.dx_trans[i] )
#                   L2 
                    self.regularizer_inter += tf.nn.l2_loss(w)

    
#       add noise bias
        singleCol_h = tf.reduce_sum( self.cx * inter_wsum, 1   )
        tmp_h=[]
        for i in range( n_hidden_list[0] ):
            with tf.variable_scope("noise"+str( i )):
                noise_b = tf.Variable(tf.random_normal([ ],\
                        stddev=math.sqrt(2.0/float(n_hidden_list[0]))))
                
                tmp_h.append( singleCol_h + noise_b ) 
        
#         tmp_h=[]
#         for i in range( n_hidden_list[0] ):
#             with tf.variable_scope("noise"+str( i )):
#                 noise_b = tf.Variable(tf.random_normal([ self.N_CONTI ],\
#                         stddev=math.sqrt(2.0/float(n_hidden_list[0]))))
#                 tmp_h.append( tf.reduce_sum( self.cx * (inter_wsum + noise_b), 1 ) ) 
        
        
        tmp_h = tf.stack( tmp_h )
        tmp_h = tf.transpose( tmp_h, [1,0])
        
        
#       dropout
#         h_inter = tf.nn.dropout(h_inter, self.keep_prob)
        
        
#       hidden layers on fusion
        with tf.variable_scope("inter_h0"):
                                    
            ortho_ini = tf.constant(self.orthogonal([n_hidden_list[0], self.N_TOTAL]), dtype = tf.float32)
            ortho_ini = tf.transpose(ortho_ini, [1,0])                        
            w = tf.Variable( ortho_ini )
#             tf.random_normal([self.N_TOTAL, n_hidden_list[0]],\
#                         stddev=math.sqrt(2.0/float(self.N_TOTAL)))) 
            b= tf.Variable(tf.zeros( [ n_hidden_list[0] ] ))
        
#           max norm constraint
#             w = tf.clip_by_norm( w, self.MAX_NORM, axes = 0) 
            
            h_inter = tf.matmul(x_concate, w)
            h_inter = h_inter + tmp_h
#           h_inter = tf.add( h_inter, b )
            
            
#           batch normalization 
            batch_m, batch_v = tf.nn.moments(h_inter,[0])
            scale = tf.Variable( tf.ones( [ n_hidden_list[0] ]) )
            beta  = tf.Variable( tf.zeros([ n_hidden_list[0] ]) )
            
            h_inter = \
            tf.nn.batch_normalization(h_inter, batch_m, batch_v, beta, scale,\
                                      self.epsilon)
            
#           nonlinear activation   
#             h_inter = tf.nn.relu( h_inter )
            h_inter = tf.maximum( 0.01*h_inter, h_inter )
            
#           L2   
            self.regularizer = tf.nn.l2_loss(w)
        
#       dropout
#         h_inter = tf.nn.dropout(h_inter, self.keep_prob)
        
        
        
        for i in range(1, self.N_HIDDEN_LAYERS):
            
            with tf.variable_scope("inter_h"+str(i)):
                                    
                ortho_ini = tf.constant(self.orthogonal([n_hidden_list[i], n_hidden_list[i-1]]), dtype = tf.float32)
                ortho_ini = tf.transpose(ortho_ini, [1,0])                        
                w = tf.Variable( ortho_ini )                    
#                 w = tf.Variable(\
#                         tf.random_normal([n_hidden_list[i-1],n_hidden_list[i]],\
#                                 stddev=math.sqrt(2.0/float(n_hidden_list[i-1])))) 
                b = tf.Variable(tf.zeros( [n_hidden_list[i]] ))
                
#               max norm constraint
#                 w = tf.clip_by_norm( w, self.MAX_NORM,axes = 0)
        
                h_inter = tf.matmul(h_inter, w)
#               h_inter = tf.add(h_inter,b)
                
                
#               batch normalization 
                batch_m, batch_v = tf.nn.moments(h_inter,[0])
                scale = tf.Variable( tf.ones( [ n_hidden_list[i] ]) )
                beta  = tf.Variable( tf.zeros([ n_hidden_list[i] ]) )
            
                h_inter = \
                tf.nn.batch_normalization(h_inter, batch_m, batch_v, beta, scale,\
                                          self.epsilon)

#               nonlinear activation  
                h_inter = tf.maximum( 0.01*h_inter, h_inter )
#                 h_inter = tf.nn.relu( h_inter )
        
#               L2
                self.regularizer += tf.nn.l2_loss(w)
    
#               dropout
#                 h_inter = tf.nn.dropout(h_inter, self.keep_prob)
        
        
#       output layer  
        h = h_inter
        
        with tf.variable_scope("output"):
            
            w = tf.Variable(tf.random_normal([n_hidden_list[self.N_HIDDEN_LAYERS-1],\
                                             self.N_CLASS],\
            stddev = math.sqrt(2.0/float(n_hidden_list[self.N_HIDDEN_LAYERS-1])))) 
            b = tf.Variable(tf.zeros( [ self.N_CLASS ] ))
            
            h = tf.matmul(h, w)
            
            h = tf.add( h, dx_wsum )
            h = tf.add( h, cooc_wsum )
            h = tf.add( h, b )
        
            self.logit = h
            
#           L2  
            self.regularizer += tf.nn.l2_loss(w) 
    
#       overall L2
        self.regularizer += \
        (self.regularizer_cooc + self.regularizer_inter + self.regularizer_wide)

        
#   initialize loss and optimization operations for training
    def train_ini(self):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                                    logits=self.logit, labels=self.y)) \
                                   + self.L2 * self.regularizer
            
        self.optimizer = \
        tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.AdadeltaOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.RMSPropOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.AdadeltaOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.cost)
        
        self.init = tf.global_variables_initializer()
        sess.run( self.init )
        
#   training on batch of data
    def train_batch(self, dx_batch,cx_batch, y_batch, keep_prob, lr):
        
        _,c = sess.run([self.optimizer,self.cost],\
                        feed_dict={self.dx:dx_batch,\
                                   self.cx:cx_batch, self.y:y_batch,\
                                   self.lr:lr,\
                                   self.keep_prob:keep_prob\
                                 })
        
        return c
        
#   initialize inference         
    def inference_ini(self):
        self.correct_prediction = tf.equal(tf.argmax(self.logit,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        
#   infer givn testing data    
    def inference(self, dx_test, cx_test, y_test, keep_prob):
        return sess.run([self.accuracy], feed_dict={self.dx:dx_test,\
                                                  self.cx:cx_test, self.y:y_test,\
                                                  self.keep_prob:keep_prob\
                                                   })
        
#   unit_test
    def test(self, dx_test, cx_test, y_test ):
        self.init = tf.global_variables_initializer()
        sess.run( self.init )
        return sess.run( [self.tmp1], \
                 feed_dict={self.dx:dx_test, self.cx:cx_test, self.y:y_test})
        
        