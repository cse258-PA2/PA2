import json
import gzip
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy
from sklearn import linear_model


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


       
## predict rating using linear regression
def predictRatingWithLinear(dataset):
        y=[int(d['rating']) for d in dataset]
        X=[feature(d) for d in dataset]
        N3=len(X)
        
        X_train=X[:(N3//11)*10]
        X_valid=X[(N3//11)*10:]
        y_train=y[:(N3//11)*10]
        y_valid=y[(N3//11)*10:]
        
        theta,residuals,rank,s=numpy.linalg.lstsq(X_train,y_train)
        
        #mse
        predict=[]
        rows=len(X_train)    
        cols=len(X_train[0])
        for i in range(rows):
            row_result=theta[0]+theta[1]*X_train[i][1]
            predict.append(row_result)
            
        differences=[(x-y)**2 for (x,y) in zip(predict,y_train)]
        MSE=sum(differences)/len(differences)
        print(str(MSE))
        return MSE
    

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

def LR(features_train,features_valid,labels_train):
    model=LogisticRegression(C=1000,random_state=1,solver='lbfgs',multi_class='multinomial',n_jobs=-1,max_iter=200)
    model.fit(features_train,labels_train)
    preds_train=model.predict(features_train)
    preds_valid=model.predict(features_valid)
    return preds_train,preds_valid


def fit_predict(dataset,modelname):
    features,labels=[],[]
    for data in dataset:
        feature=[data['height'],data['weight'],data['bust size'],data['age'],data['size']]
        body_type=[int(i) for i in list(data['body type'])]
        feature+=body_type
        features.append(feature)
        labels.append(data['fit'])

    features_train,features_valid,labels_train,labels_valid=train_test_split(features,labels,test_size=1/10,random_state=1,shuffle=True)

    if modelname=="LogisticRegression":
        preds_train,preds_valid=LR(features_train,features_valid,labels_train)


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

    user_reviews=defaultdict(list)  #reviews of each user
    item_reviews=defaultdict(list)  #reviews of each item

    for data in dataset:
        user_reviews["user_id"].append(data)
        item_reviews['item_id'].append(data)

    print('Read dataset complete')

#-------------------------plot statistics of dataset-----------------------------
    #plot_dataset_statistics(dataset)

#-------------------------rating prediction--------------------------------------
    a=predictRatingWithLinear(dataset)
    
#-------------------------fit prediction-----------------------------------------
    fit_predict(dataset,"LogisticRegression")




