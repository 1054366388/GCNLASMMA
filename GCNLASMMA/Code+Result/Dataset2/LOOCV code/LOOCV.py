#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import *
from model import GCNModel
from opt import Optimizer
import numpy as np
import numpy.linalg as LA
import scipy.linalg as LG
import time

def solve_l1l2(W,lambda1):
    nv=W.shape[1]
    F=W.copy()
    for p in range(nv):
        nw=LA.norm(W[:,p],"fro")
        if nw>lambda1:
            F[:,p]=(nw-lambda1)*W[:,p]/nw
        else:F[:,p]=np.zeros((W[:,p].shape[0],1))
    return F

def PredictScore(train_drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_drug_dis_matrix, drug_matrix, dis_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_drug_dis_matrix.sum()
    X = constructNet(train_drug_dis_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_drug_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_drug_dis_matrix.shape[0], name='LAGCN')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_drug_dis_matrix.shape[0], num_v=train_drug_dis_matrix.shape[1], association_nam=association_nam)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res

time1 = time.time()
a=np.mat(np.zeros((286,39)))#Define a 286*39 matrix of zeros
AA=a.astype(int)#Convert matrix AA to integer

#Read the known SM-miRNA associations
b=np.loadtxt('known SM-miRNA interaction.txt')
B=b.astype(int)

#Change the corresponding position element of matrix AA to 1 to obtain the adjacency matrix AA
for x in B:
    AA[x[0]-1,x[1]-1]=1

#Read the integrated similarity of SM
SSM=np.mat(np.loadtxt('Integrated Similarity of SM.txt'))

#Read the integrated similarity of miRNA
SMR=np.mat(np.loadtxt('Integrated similarity of miRNA.txt'))

#Normalize the SSM
SSM1=np.mat(SSM.copy())
for nn1 in range(39):
    for nn2 in range(39):
        SSM[nn1,nn2]=SSM[nn1,nn2]/(np.sqrt(np.sum(SSM1[nn1,:]))*np.sqrt(np.sum(SSM1[nn2,:])))

#Normalize the SMR
SMR1=np.mat(SMR.copy())
for mm1 in range(286):
    for mm2 in range(286):
        SMR[mm1,mm2]=SMR[mm1,mm2]/(np.sqrt(np.sum(SMR1[mm1,:]))*np.sqrt(np.sum(SMR1[mm2,:])))

#LOOCV
for y in range(664):
    A=np.mat(AA.copy())#Copy the original adjacency matrix AA to A
    i=B[y,0]-1
    j=B[y,1]-1
    A[i,j]=0#Remove one of the SM-miRNA associations in turn

#Matrix decomposition
    alpha = 0.1
    J = np.mat(np.zeros((39, 39)))
    X = np.mat(np.zeros((39, 39)))
    E = np.mat(np.zeros((286, 39)))
    Y1 = np.mat(np.zeros((286, 39)))
    Y2 = np.mat(np.zeros((39, 39)))
    mu = 10 ** -4
    max_mu = 10 ** 10
    rho = 1.1
    epsilon = 10 ** 10
    while True:
        [U, sigma1, V] = LG.svd(X + Y2 / mu, lapack_driver='gesvd')
        G = [sigma1[k] for k in range(len(sigma1)) if sigma1[k] > 1 / mu]
        svp = len(G)
        if svp >= 1:
            sigma1 = sigma1[0:svp] - 1 / mu
        else:
            sigma1 = [0]
            svp = 1
        J = np.mat(U[:, 0:svp]) * np.mat(np.diag(sigma1)) * np.mat(V[0:svp, :])
        ATA = A.T * A
        X = (ATA + np.eye(39)).I * (ATA - A.T * E + J + (A.T * Y1 - Y2) / mu)
        temp1 = A - A * X
        E = solve_l1l2(temp1 + Y1 / mu, alpha / mu)
        Y1 = Y1 + mu * (temp1 - E)
        Y2 = Y2 + mu * (X - J)
        mu = min(rho * mu, max_mu)
        if LA.norm(temp1 - E, np.inf) < epsilon and LA.norm(X - J, np.inf) < epsilon: break
    P= A * X

    # GCN
    miRNA_sim = SMR
    SM_sim = SSM
    SM_miRNA_matrix = P
    epoch = 600
    emb_dim = 64
    lr = 0.00725
    adjdp = 0.6
    dp = 0.4
    simw = 6
    miRNA_matrix = miRNA_sim * simw
    SM_matrix = SM_sim * simw
    train_matrix = np.matrix(SM_miRNA_matrix, copy=True)
    circle_time = 0
    miRNA_len = SM_miRNA_matrix.shape[0]
    SM_len = SM_miRNA_matrix.shape[1]
    SM_miRNA_res = PredictScore(train_matrix, miRNA_matrix, SM_matrix, circle_time, epoch, emb_dim, dp, lr,
                                adjdp)
    S = np.mat(SM_miRNA_res.reshape(miRNA_len, SM_len))
    print('--------------------------------------')


#Calculate the score of elements that change from 1 to 0
    S1=S[i,j]
#global LOOOCV
    list1=[S1]#Define a list1 contains scores of element which change from 1 to 0
    for mi in range(286):
        for di in range(39):
            if AA[mi,di]==0:
                SG=S[mi,di]
                list1.append(SG)#Add the scores of the remaining 0 elements in AA to list1
            else:
                continue
    list2=sorted(list1,reverse=True)#Sort the score array list1 from largest to smallest
    arrayp1=np.array(list2)
    locationp1=np.average(np.where(arrayp1==S1))#Find the ranking position of the score from 1 to 0 elements and find the average

#local LOOCV SM-fixed
    list3=[S1]  #Define a list3 contains scores of element which change from 1 to 0
    for u in range(286):
        if AA[u,j]==0:
            SL1=S[u,j]
            list3.append(SL1)#Add the scores of the remaining 0 elements in AA to list3
        else:
            continue
    list4=sorted(list3,reverse=True)#Sort the score array list3 from largest to smallest
    arrayp2=np.array(list4)
    locationp2=np.average(np.where(arrayp2==S1))#Find the ranking position of the score from 1 to 0 elements and find the average
    print('===============================================================================================')
    print('global=',locationp1+1)
    print('local=',locationp2+1)
    Global = open('../LOOCV result/Global.txt','a+')
    Global.writelines(['\n',str(locationp1+1)])
    Global.close()
    Locall=open('../LOOCV result/SM-fixed_Local.txt','a+')
    Locall.writelines(['\n',str(locationp2+1)])
    Locall.close()

#local LOOCV miRNA-fixed
    list5 = [S1]  #Define a list5 contains scores of element which change from 1 to 0
    for v in range(39):
        if AA[i, v] == 0:
            SL2 = S[i, v]
            list5.append(SL2)  #Add the scores of the remaining 0 elements in AA to list5
        else:
            continue
    list6 = sorted(list5, reverse=True)  #Sort the score array list5 from largest to smallest
    arrayp3 = np.array(list6)
    locationp3 = np.average(np.where(arrayp3 == S1))  #Find the ranking position of the score from 1 to 0 elements and find the average
    print('===============================================================================================')
    print('local=', locationp3 + 1)
    Locall = open('../LOOCV result/miRNA-fixed_Local.txt', 'a+')
    Locall.writelines(['\n', str(locationp3 + 1)])
    Locall.close()





