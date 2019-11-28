import json
import gzip
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy
from sklearn import linear_model


figs_path="/mnt/c/Users/AlbertPi/Desktop/"

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
    feat=[1,int(datum['whetherFit']),int(len(datum['review_text'])),]
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
        rows=len(X_valid)    
        cols=len(X_valid[0])
        for i in range(rows):
            row_result=theta[0]+theta[1]*X_valid[i][1]
            predict.append(row_result)
            
        differences=[(x-y)**2 for (x,y) in zip(predict,y_valid)]
        MSE=sum(differences)/len(differences)
        print(str(MSE))
        return MSE
    


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
            data['age']=int(data['age'])
            data['rating']=int(data['rating'])
            data['weight']=int(data['weight'][:-3])
            dataset.append(data)

    user_reviews=defaultdict(list)  #reviews of each user
    item_reviews=defaultdict(list)  #reviews of each item

    for data in dataset:
        user_reviews["user_id"].append(data)
        item_reviews['item_id'].append(data)

    #plot statistics of dataset
    #plot_dataset_statistics(dataset)
    a=predictRatingWithLinear(dataset)
    
