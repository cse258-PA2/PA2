import json
import gzip
import numpy as np
import string
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import linear_model
import torch
from torch import nn
import torch.optim as optim

figs_path="/mnt/c/Users/AlbertPi/Desktop/"
body_type_convertion_mod='onehot'  #'onehot' or 'integer'
wordBag_size=1000

#plot statistics of dataset
def plot_dataset_statistics(dataset):
    statistics_dic={}
    statistics_dic['fit']=defaultdict(int)
    statistics_dic['bust size']=defaultdict(int)
    statistics_dic['weight']=defaultdict(int)
    statistics_dic['rating']=defaultdict(int)
    statistics_dic['rented for']=defaultdict(int)
    statistics_dic['body type']=defaultdict(int)
    statistics_dic['category']=defaultdict(int)
    statistics_dic['height']=defaultdict(int)
    statistics_dic['size']=defaultdict(int)
    statistics_dic['age']=defaultdict(int)

    for data in dataset:
        for head in data:
            if head!='user_id' and head!='item_id' and head!='review_text' and head!='review_summary' and head!='review_date':
                statistics_dic[head][data[head]]+=1      

    for head in statistics_dic:
        fig, ax = plt.subplots(1, 1)
        plt.figure(figsize=(16,9))
        tmp_lst=list(zip(statistics_dic[head].keys(),statistics_dic[head].values()))
        tmp_lst.sort()
        values,keys=[item[1] for item in tmp_lst],[item[0] for item in tmp_lst]
        fig=plt.bar(range(len(values)),values)
        stride=1
        plt.xticks(range(len(values))[::stride],keys[::stride])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(stride))
        plt.xlabel(head)
        plt.ylabel("Number of samples")
        # plt.title(head+" statistics")
        plt.savefig(figs_path+head+"_statistics.png",dpi=300)

def bust_size_convertion(bust_size):
    lower_size=int(bust_size[:2])
    cup_size=bust_size[2:]
    cup_size_dic={'aa':1,'a':2,'b':3,'c':4,'d':5,'d+':5.5,'dd':6,'ddd/e':7,'f':9,'g':11,'h':13,'i':15,'j':17}
    return lower_size+cup_size_dic[cup_size]

def body_type_convertion(body_type):
    if body_type_convertion_mod=='onehot':
        dic={'apple':'1000000','athletic':'0100000','full bust':'0010000','hourglass':'0001000','pear':'0000100','petite':'0000010','straight & narrow':'0000001'}
    else:
        dic={'apple':'1','athletic':'2','full bust':'3','hourglass':'4','pear':'5','petite':'6','straight & narrow':'7'}
    return dic[body_type]

##------------------------------------------------------------
#                   Rating Prediction
##------------------------------------------------------------

def commonWords_generate(dataset,begin_index,end_index):
    wordCount_dict=defaultdict(int)
    punct = string.punctuation

    for data in dataset:
        review=data['review_text'].lower()
        review=[char for char in review if (char not in punct)]
        review=''.join(review)
        data['review_text']=review
        words=review.strip().split()
        for word in words:
            wordCount_dict[word]+=1
    
    wordANDcount=[(wordCount_dict[word],word) for word in wordCount_dict]
    wordANDcount.sort(reverse=True)
    commonWords=[sample[1] for sample in wordANDcount[begin_index:end_index]]
    commonWordsID=dict(zip(commonWords,range(len(commonWords))))
    return commonWordsID

def freqVec_generate(review,commonWordsID):
    words=review.strip().split()
    vector=[0]*len(commonWordsID)
    for word in words:
        if word in commonWordsID:
            vector[commonWordsID[word]]+=1
    return vector

def MSE(preds,labels):
    differences=[(x-y)**2 for (x,y) in zip(preds,labels)]
    MSE=sum(differences)/len(differences)
    return MSE

def Naive_Average(labels_train,labels_valid):
    pred=sum(labels_train)/len(labels_train)
    preds_train=[pred]*len(labels_train)
    preds_valid=[pred]*len(labels_valid)

    MSE_train=MSE(preds_train,labels_train)
    MSE_valid=MSE(preds_valid,labels_valid)
    print("On train set, MSE = " + str(MSE_train)+", On valid set, MSE = " + str(MSE_valid))
            
    return MSE_valid

def Linear_Regression(features_train,features_valid,labels_train,labels_valid):
        model=LinearRegression()
        features_train=np.array(features_train)
        features_valid=np.array(features_valid)
        model.fit(features_train[:,2:],labels_train)
        
        preds_train=model.predict(features_train[:,2:])
        preds_valid=model.predict(features_valid[:,2:])
        MSE_train=MSE(preds_train,labels_train)
        MSE_valid=MSE(preds_valid,labels_valid)
        print("On train set, MSE = " + str(MSE_train)+", On valid set, MSE = " + str(MSE_valid))
            
        return MSE_valid

def MSE_loss(preds,labels):
    return ((preds-labels)**2).mean()

class LatentFactorModel(nn.Module):
    def __init__(self,num_users,num_items,K,alpha):
        super(LatentFactorModel,self).__init__()
        self.alpha=nn.Parameter(torch.tensor([alpha],dtype=torch.float))
        self.userBias=nn.Parameter(torch.zeros(num_users,1))
        self.itemBias=nn.Parameter(torch.zeros(1,num_items))
        self.gamma_user=nn.Parameter(torch.zeros(num_users,K))
        self.gamma_item=nn.Parameter(torch.zeros(K,num_items))
        self.fit_layer=nn.Linear(3+wordBag_size,1)
    
    def forward(self,users,items,feature):
        rating_preds=self.alpha+self.userBias[users,0]+self.itemBias[0,items]+(self.gamma_user[users,:]*torch.transpose(self.gamma_item[:,items],0,1)).sum()+self.fit_layer(feature).squeeze(1)
        # rating_preds=self.alpha+self.userBias[users,0]+self.itemBias[0,items]+self.fit_layer(fit_info).squeeze(1)
        return rating_preds

class DenseNet(nn.Module):
    def __init__(self,num_users,num_items,K,num_Hidden):
        super(DenseNet,self).__init__()
        self.gamma_users=nn.Parameter(torch.randn(num_users,K))
        self.gamma_items=nn.Parameter(torch.randn(num_items,K))
        self.fullyConnect1=nn.Linear(2*K+3,num_Hidden)
        self.activation1=nn.ReLU()
        self.fullyConnect2=nn.Linear(num_Hidden,1)
    
    def forward(self,users,items,fit_info):
        users_vector=self.gamma_users[users,:]
        items_vector=self.gamma_items[items,:]
        vector=torch.cat([users_vector,items_vector,fit_info],1)
        # vector=fit_info
        activated_vector=self.activation1(self.fullyConnect1(vector))
        # activated_vector=self.fullyConnect1(vector)
        return self.fullyConnect2(activated_vector).squeeze(1)

def rating_prediction_training(rating_features_train,rating_features_valid,rating_labels_train,rating_labels_valid,num_users,num_items,modelname):
    alpha=sum(rating_labels_train)/len(rating_labels_train)
    rating_features_train=torch.tensor(rating_features_train,dtype=torch.long)
    rating_features_valid=torch.tensor(rating_features_valid,dtype=torch.long)
    rating_labels_train=torch.tensor(rating_labels_train,dtype=torch.float32)
    rating_labels_valid=torch.tensor(rating_labels_valid,dtype=torch.float32)

    if modelname=='LatentFactorModel':
        model=LatentFactorModel(num_users,num_items,5,alpha)
        learning_rate=5e-1
        optimizer=optim.SGD(model.parameters(),lr=learning_rate,momentum=0.8,weight_decay=5e-4)
    elif modelname=='DenseNet':
        model=DenseNet(num_users,num_items,5,30)
        learning_rate=1e-2
        optimizer=optim.SGD(model.parameters(),lr=learning_rate,momentum=0.8,weight_decay=5e-5)
        # optimizer=optim.Adam(model.parameters(),lr=learning_rate,weight_decay=5e-3)

    loss_func=MSE_loss

    train_epochs=20000
    for epoch in range(train_epochs):

        optimizer.zero_grad()
        rating_preds_train=model(rating_features_train[:,0],rating_features_train[:,1],rating_features_train[:,2:].float())
        loss_train=loss_func(rating_preds_train,rating_labels_train)
        loss_train.backward()

        optimizer.step()

        rating_preds_valid=model(rating_features_valid[:,0],rating_features_valid[:,1],rating_features_valid[:,2:].float())
        loss_valid=loss_func(rating_preds_valid,rating_labels_valid)
        if epoch<5 or epoch%100==0:
            print("Epoch: %d, train loss is: %f, validation loss is: %f" %(epoch,float(loss_train),float(loss_valid)))
    return loss_valid

def rating_prediction(dataset,modelname):

    dataset_train,dataset_valid=train_test_split(dataset,test_size=1/5,random_state=1)
    commonWordsID=commonWords_generate(dataset_train,0,0+wordBag_size)

    user_set,item_set=set(),set()
    for data in dataset:
        user_set.add(data['user_id'])
        item_set.add(data['item_id'])
    users,items=list(user_set),list(item_set)
    userIndex_dic,itemIndex_dic={},{}
    userIndex_dic=dict(zip(users,range(len(users))))
    itemIndex_dic=dict(zip(items,range(len(items))))

    rating_features,rating_labels,wordFeq_vectors=[],[],[]
    fit_dic={'fit':[1,0,0],'large':[0,1,0],'small':[0,0,1]}
    for data in dataset:
        rating_features.append([userIndex_dic[data['user_id']],itemIndex_dic[data['item_id']]]+fit_dic[data['fit']]+freqVec_generate(data['review_text'],commonWordsID))
        rating_labels.append(data['rating'])
    
    rating_features_train,rating_features_valid,rating_labels_train,rating_labels_valid=train_test_split(rating_features,rating_labels,test_size=1/5,random_state=1)
    
    if modelname=='NaiveAverage':
        MSE=Naive_Average(rating_labels_train,rating_labels_valid)
    elif modelname=="LinearRegression":
        MSE=Linear_Regression(rating_features_train,rating_features_valid,rating_labels_train,rating_labels_valid)
    elif modelname=='LatentFactorModel':
        MSE=rating_prediction_training(rating_features_train,rating_features_valid,rating_labels_train,rating_labels_valid,len(users),len(items),'LatentFactorModel')
    elif modelname=='DenseNet':
        MSE=rating_prediction_training(rating_features_train,rating_features_valid,rating_labels_train,rating_labels_valid,len(users),len(items),'DenseNet')
    
    print("MSE of "+modelname+" on validation set is: %f" %MSE)

##------------------------------------------------------------
#                   Fit Prediction
##------------------------------------------------------------

def classification_statics(preds,labels):
    cm=metrics.confusion_matrix(labels,preds)
    template="{0:10}{1:8}{2:8}{3:8}"
    print(template.format('predict','fit','large','small'))
    print('actual')
    print(template.format('fit',str(cm[0][0]),str(cm[0][1]),str(cm[0][2])))
    print(template.format('large',str(cm[1][0]),str(cm[1][1]),str(cm[1][2])))
    print(template.format('small',str(cm[2][0]),str(cm[2][1]),str(cm[2][2])))
    print('\n-----------------------------------------')
    print(metrics.classification_report(labels,preds,digits=3))

def PCA_Reduction(features_train,features_valid,components):
    pca = PCA(n_components=components)
    pca.fit(features_train)
    features_train = pca.transform(features_train)
    features_valid = pca.transform(features_valid)
    return features_train,features_valid

def Logistic_Regression(features_train,features_valid,labels_train):
    model=LogisticRegression(C=0.1,random_state=1,solver='lbfgs',multi_class='multinomial',n_jobs=-1,max_iter=2000,
                             class_weight={'fit':0.3, 'small':1, 'large':1.2})
    model.fit(features_train,labels_train)
    preds_train=model.predict(features_train)
    preds_valid=model.predict(features_valid)
    return preds_train,preds_valid

def SVM(features_train,features_valid,labels_train):
    model=SVC(kernel='sigmoid',C=0.1,random_state=1,gamma='auto',max_iter=2000)
    model.fit(features_train,labels_train)
    preds_train=model.predict(features_train)
    preds_valid=model.predict(features_valid)
    return preds_train,preds_valid

def RandomForest(features_train,features_valid,labels_train):
    model=RandomForestClassifier(n_estimators=500, max_depth=3,random_state=1)
    model.fit(features_train,labels_train)
    preds_train=model.predict(features_train)
    preds_valid=model.predict(features_valid)
    return preds_train,preds_valid

def GradientBoosting(features_train,features_valid,labels_train):
    model=GradientBoostingClassifier(n_estimators=500,random_state=1)
    model.fit(features_train,labels_train)
    preds_train=model.predict(features_train)
    preds_valid=model.predict(features_valid)
    return preds_train,preds_valid

def fit_prediction(dataset,modelname):
    features,labels=[],[]
    for data in dataset:
        feature=[data['height'],data['weight'],data['bust size'],data['age'],data['size']]
        body_type=[int(i) for i in list(data['body type'])]
        feature+=body_type
        features.append(feature)
        labels.append(data['fit'])

    # take 0.6 of dataset for actual use
    features,_,labels,_=train_test_split(features,labels,train_size=1/2,random_state=1)

    features_train,features_valid,labels_train,labels_valid=train_test_split(features,labels,test_size=1/5,random_state=1)

    # perform pca
    features_train,features_valid = PCA_Reduction(features_train, features_valid, 8)

    if modelname=="LogisticRegression":
        preds_train,preds_valid=Logistic_Regression(features_train,features_valid,labels_train)

    elif modelname=="SVM":
        preds_train,preds_valid=SVM(features_train,features_valid,labels_train)

    elif modelname=="RandomForest":
        preds_train, preds_valid=RandomForest(features_train, features_valid, labels_train)

    elif modelname=='GradientBoosting':
        preds_train, preds_valid=GradientBoosting(features_train, features_valid, labels_train)

    print('-----------------------------------------')
    print('On Train Set')
    print('-----------------------------------------')
    classification_statics(preds_train,labels_train)

    print('-----------------------------------------')
    print('On Validation Set')
    print('-----------------------------------------')
    classification_statics(preds_valid,labels_valid)

def category_size_distribution(dataset):
    size_dic=defaultdict(set)
    count_dic=defaultdict(int)
    for data in dataset:
        size_dic[data['category']].add(data['size'])
        count_dic[data['category']]+=1
    
    for key in size_dic:
        print(key+': '+str(count_dic[key])+str(size_dic[key]))


#Possible tasks: 1. predict whether fit  2. predict rating 3. whether A would rent B
if __name__ == "__main__":
    dataset=[]
    null=None #fix small issue of json reading
    with gzip.open("cloth_data.json.gz",'rt') as f:
        for line in f:
            data=json.loads(line)
            if data['rating']==None or len(data)!=15:  #If it doesn't have all attributes or rating is None, abort thus sample  
                continue        
            data['age']=int(data['age'])
            data['rating']=int(data['rating'])
            data['weight']=int(data['weight'][:-3])
            height_tmp=data['height'].strip('\"').split('\'')
            data['height']=0.3*float(height_tmp[0])+2.54*float(height_tmp[1])/100
            data['bust size']=bust_size_convertion(data['bust size'])
            data['body type']=body_type_convertion(data['body type'])
            dataset.append(data)

    print('Read dataset complete')

    # category_size_distribution(dataset)

#-------------------------plot statistics of dataset-----------------------------
    #plot_dataset_statistics(dataset)

#-------------------------rating prediction--------------------------------------
    rating_prediction(dataset,'NaiveAverage')
    # rating_prediction(dataset,'LinearRegression')
    # rating_prediction(dataset,'LatentFactorModel')
    # rating_prediction(dataset,'DenseNet')
    
#-------------------------fit prediction-----------------------------------------
    # print('-----------------------------------------')
    # print('Using LR')
    # print('-----------------------------------------')
    # fit_prediction(dataset,"LogisticRegression")
    #
    # print('-----------------------------------------')
    # print('Using SVM')
    # print('-----------------------------------------')
    # fit_prediction(dataset,"SVM")
    #
    # print('-----------------------------------------')
    # print('Using Random Forest')
    # print('-----------------------------------------')
    # fit_prediction(dataset,"RandomForest")
    #
    # print('-----------------------------------------')
    # print('Using Gradient Boosting')
    # print('-----------------------------------------')
    # fit_prediction(dataset,"GradientBoosting")




