import json
import gzip
import numpy
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
import torch
from torch import nn
import torch.optim as optim


figs_path="/mnt/c/Users/AlbertPi/Desktop/"
body_type_convertion_mod='onehot'  #'onehot' or 'integer'

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

## use to record features used in linear regression
def feature(datum):
    feat=[1,int(datum['whetherFit'])]
    return feat

def MSE(preds,labels):
    differences=[(x-y)**2 for (x,y) in zip(preds,labels)]
    MSE=sum(differences)/len(differences)
    return MSE


       
## predict rating using linear regression
def Linear_Regression(dataset):
        labels=[int(d['rating']) for d in dataset]
        X=[feature(d) for d in dataset]
        X_train,X_valid,labels_train,labels_valid=train_test_split(X,labels,test_size=1/5,random_state=1)
        model=LinearRegression()
        model.fit(X_train,labels_train)
        
        preds_train=model.predict(X_train)
        preds_valid=model.predict(X_valid)
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
        self.fit_layer=nn.Linear(3,1)
    
    def forward(self,users,items,fit_info):
        # rating_preds=self.alpha+self.userBias[users,0]+self.itemBias[0,items]+(self.gamma_user[users,:]*torch.transpose(self.gamma_item[:,items],0,1)).sum()
        rating_preds=self.alpha+self.userBias[users,0]+self.itemBias[0,items]+self.fit_layer(fit_info).squeeze(1)
        # print(self.fit_layer(fit_info).shape)
        return rating_preds

class DenseNet(nn.Module):
    def __init__(self,num_users,num_items,K,num_Hidden):
        super(DenseNet,self).__init__()
        self.gamma_users=nn.Parameter(torch.randn(num_users,K))
        self.gamma_items=nn.Parameter(torch.randn(num_items,K))
        self.layer1=nn.Linear(2*K,num_Hidden)
        self.layer1_activate=nn.ReLU()
        self.layer2=nn.Linear(num_Hidden,1)
    
    def forward(self,users,items):
        users_vector=self.gamma_users[users,:]
        items_vector=self.gamma_items[items,:]
        vector=torch.cat([users_vector,items_vector],1)
        activated_vector=self.layer1_activate(self.layer1(vector))
        # activated_vector=self.layer1(vector)
        return self.layer2(activated_vector).squeeze(1)
        

def LFM_prediction(rating_dataset_train,rating_dataset_valid,rating_labels_train,rating_labels_valid,num_users,num_items):
    alpha=sum(rating_labels_train)/len(rating_labels_train)

    model=LatentFactorModel(num_users,num_items,20,alpha)
    # model=DenseNet(num_users,num_items,4,25)
    loss_func=MSE_loss
    learning_rate=1e-1
    optimizer=optim.SGD(model.parameters(),lr=learning_rate,momentum=0.8,weight_decay=5e-4)
    # optimizer=optim.Adam(model.parameters(),lr=learning_rate,weight_decay=5e-3)

    train_epochs=20000
    for epoch in range(train_epochs):
        # print(float(model.alpha))
        optimizer.zero_grad()
        rating_preds_train=model(rating_dataset_train[:,0],rating_dataset_train[:,1],rating_dataset_train[:,2:].float())
        loss_train=loss_func(rating_preds_train,rating_labels_train)
        loss_train.backward()

        optimizer.step()

        rating_preds_valid=model(rating_dataset_valid[:,0],rating_dataset_valid[:,1],rating_dataset_valid[:,2:].float())
        loss_valid=loss_func(rating_preds_valid,rating_labels_valid)
        if epoch<5 or epoch%100==0:
            print("Epoch: %d, train loss is: %f, validation loss is: %f" %(epoch,float(loss_train),float(loss_valid)))
    return loss_valid

        



def rating_prediction(dataset,modelname):
    user_set=set()
    item_set=set()
    for data in dataset:
        user_set.add(data['user_id'])
        item_set.add(data['item_id'])
    users=list(user_set)
    items=list(item_set)
    userIndex_dic={}
    itemIndex_dic={}
    cnt=0
    for user in users:
        userIndex_dic[user]=cnt
        cnt+=1
    cnt=0
    for item in items:
        itemIndex_dic[item]=cnt
        cnt+=1

    rating_dataset=[]
    rating_labels=[]
    fit_dic={'fit':[1,0,0],'large':[0,1,0],'small':[0,0,1]}
    for data in dataset:
        rating_dataset.append([userIndex_dic[data['user_id']],itemIndex_dic[data['item_id']]]+fit_dic[data['fit']])
        rating_labels.append(data['rating'])
        
    
    rating_dataset_train,rating_dataset_valid,rating_labels_train,rating_labels_valid=train_test_split(rating_dataset,rating_labels,test_size=1/5)
    rating_dataset_train=torch.tensor(rating_dataset_train,dtype=torch.long)
    rating_dataset_valid=torch.tensor(rating_dataset_valid,dtype=torch.long)
    rating_labels_train=torch.tensor(rating_labels_train,dtype=torch.float32)
    rating_labels_valid=torch.tensor(rating_labels_valid,dtype=torch.float32)
    
    MSE=Linear_Regression(dataset)
    if modelname=="LinearRegression":
        MSE=Linear_Regression(dataset)
    elif modelname=='LatentFactorModel':
        MSE=LFM_prediction(rating_dataset_train,rating_dataset_valid,rating_labels_train,rating_labels_valid,len(users),len(items))
    
    print("MSE of "+modelname+" on validation set is: %f" %MSE)
    

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

def Logistic_Regression(features_train,features_valid,labels_train):
    model=LogisticRegression(C=1000,random_state=1,solver='lbfgs',multi_class='multinomial',n_jobs=-1,max_iter=200)
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

    features_train,features_valid,labels_train,labels_valid=train_test_split(features,labels,test_size=1/10)

    if modelname=="LogisticRegression":
        preds_train,preds_valid=Logistic_Regression(features_train,features_valid,labels_train)


    print('-----------------------------------------')
    print('On Train Set')
    print('-----------------------------------------')
    classification_statics(preds_train,labels_train)

    print('-----------------------------------------')
    print('On Validation Set')
    print('-----------------------------------------')
    classification_statics(preds_valid,labels_valid)
    


#Possible tasks: 1. predict whether fit  2. predict rating 3. whether A would rent B
if __name__ == "__main__":
    dataset=[]
    null=None #fix small issue of json reading
    with gzip.open("cloth_data.json.gz",'rt') as f:
        for line in f:
            data=json.loads(line)
            if data['rating']==None or len(data)!=15:  #If it doesn't have all attributes or rating is None, abort thus sample  
                continue
            if data['fit']=='fit':
                data['whetherFit']=1
            else:
                data['whetherFit']=0         
            if 'perfect' in data['review_summary'] or 'glamourous' in data['review_summary'] or 'love' in data['review_summary'] or 'cute' in data['review_summary'] or 'compliments' in data['review_summary'] or 'confident' in data['review_summary'] or 'awesome' in data['review_summary'] or 'comfortable' in data['review_summary'] or 'fashion' in data['review_summary'] or 'trendy' in data['review_summary'] or 'great' in data['review_summary'] or 'best' in data['review_summary'] or 'gorgeous' in data['review_summary'] or 'beautiful' in data['review_summary'] or 'recommend' in data['review_summary'] or 'fit' in data['review_summary'] or 'fun' in data['review_summary']:
                data['good_review']=1
            else:
                data['good_review']=0
            data['age']=int(data['age'])
            data['rating']=int(data['rating'])
            data['weight']=int(data['weight'][:-3])
            height_tmp=data['height'].strip('\"').split('\'')
            data['height']=float(height_tmp[0])+0.1*float(height_tmp[1])
            data['bust size']=bust_size_convertion(data['bust size'])
            data['body type']=body_type_convertion(data['body type'])
            dataset.append(data)



    print('Read dataset complete')

#-------------------------plot statistics of dataset-----------------------------
    #plot_dataset_statistics(dataset)

#-------------------------rating prediction--------------------------------------
    rating_prediction(dataset[:],'LatentFactorModel')
    
#-------------------------fit prediction-----------------------------------------
    # fit_prediction(dataset,"LogisticRegression")




