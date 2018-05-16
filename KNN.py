import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
def knncls():
    """
     K近邻算法预测用户入住的位置
    :return:None
    """
    #读取数据
    data = pd.read_csv("train.csv")
    #print(data.head(10))

    #处理数据
    #1,缩小数据,查询数据筛选
    data = data.query('x>1.0&x<2.25&y>2.5&y<2.75')
    #2,处理时间戳的数据
    time_value = pd.to_datetime(data['time'],unit='s')
    #print(time_value)
    # 将时间的格式转化为字典的格式
    time_value = pd.DatetimeIndex(time_value)
    #print(time_value)


    #构造特征,加入day,hour,weekday特征

    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday
    # data.loc[:,'day'] = time_value.day
    # data.loc[:,'hour'] = time_value.hour
    # data.loc[:,'weekday'] = time_value.weekday
    #print(data.head(10))
    #删除时间戳的数据
    data = data.drop(['time'],axis=1)
    #print(data.head(10))

    #将签到的次数小于n的位置删除，这里我设定n=3
    place_count = data.groupby('place_id').count()#以place_id分组，统计相同的次数
    #print(place_count)
    tf = place_count[place_count.row_id>3].reset_index()
    #print(tf)
    data = data[data['place_id'].isin(tf['place_id'])]
    #print(data)

    #取出数据中的特征值和目标值
    y = data['place_id']
    #print(y)
    x = data.drop(['place_id'],axis=1)
    x = data.drop(['row_id'],axis=1)#将无关的row_id删除，提高预测数据的准确率
    #print(x.head(10))

    #进行数据的分割，分割训练集和测试集
    x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.25)
    print(x_test)

    #将数据标准化,惊奇地发现不对数据集进行标准化居然达到的准确率最高
    # std = StandardScaler()
    # x_train = std.fit_transform(x_train)
    # x_test = std.transform(x_test)
    knn = KNeighborsClassifier(n_neighbors=5)
    #喂数据
    knn.fit(x_train,y_train)
    y_predict = knn.predict(x_test)
    print("预测的位置为：",y_predict)
    print("预测的准确率为：",knn.score(x_test,y_test))
    return None
if __name__ == '__main__':
    knncls()