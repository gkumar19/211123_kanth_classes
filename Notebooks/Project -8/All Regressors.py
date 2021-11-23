import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import explained_variance_score 
import statsmodels.formula.api as smf
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn import preprocessing

data=pd.read_csv("Downloads\\Automobile price data _Raw_.csv")
data.columns=['symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration',
       'num_of_doors', 'body_style', 'drive_wheels', 'engine_location',
       'wheel_base', 'length', 'width', 'height', 'curb_weight', 'engine_type',
       'num_of_cylinders', 'engine_size', 'fuel_system', 'bore', 'stroke',
       'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg',
       'highway_mpg', 'price']

data.replace('?',np.nan,inplace=True)
#we cant find mean for 'object' datatype so converting the dtype to 'float64'
data.normalized_losses=data.normalized_losses.astype('float64')
data.dtypes
#filling nulls of  normalized_losses column  with mean
data['normalized_losses'].fillna((data['normalized_losses'].mean()), inplace=True)
data['normalized_losses'].mean()
data.isnull().sum()
data.normalized_losses.value_counts()

#changing the dtype of symboling 
data.symboling=data.symboling.astype('category')

#imputing with mode 
data.num_of_doors.value_counts()
data.num_of_doors.mode()
data['num_of_doors'].fillna((data['num_of_doors'].mode()[0]), inplace=True)
#imputing the nan values in 'bore' column with the mode
data.bore.fillna((data.bore.mode()[0]),inplace=True)
data.bore=data.bore.astype("float64")
#imputing the nan values in 'stroke' column with the mode
data.stroke.fillna((data.stroke.mode()[0]),inplace=True)
data.stroke.value_counts()
data.stroke= data.stroke.astype("float64")

#imputing the nan values in 'horsepower' column with the mode
data.horsepower.fillna((data.horsepower.mode()[0]),inplace=True)
data.horsepower=data.horsepower.astype("int64")
#imputing the nan values in 'peak_rpm' column with the mode
data.peak_rpm.fillna((data.peak_rpm.mode()[0]),inplace=True)
data.peak_rpm= data.peak_rpm.astype("int64")
#imputing the nan values in 'price' column with the mean
data.price=data.price.astype("float64")
data.price.fillna(data.price.mean(),inplace=True)

#univariate analysis
sns.boxplot(data.normalized_losses)
sns.boxplot(data.wheel_base)
sns.boxplot(data.length)
sns.boxplot(data.width)
sns.boxplot(data.height)
sns.boxplot(data.curb_weight)
sns.boxplot(data.engine_size)
sns.boxplot(data.bore)
sns.boxplot(data.stroke)

#bivariate analysis
correlation=data.corr()


x=data.iloc[:,0:25]
y=data.iloc[:,25:26]

x_dummies=pd.get_dummies(x,drop_first=True)


xtrain, xtest, ytrain, ytest= train_test_split(x_dummies,y, test_size=.3, random_state=234)
##############################################################################################
#building model using xgboost algorithm
xgb2 = xgb.XGBRegressor(n_estimators=12,learning_rate=0.2) 
xgb2.fit(xtrain,ytrain) 
#predicting output for training data
ytrain["predict_xgboost"]=xgb2.predict(xtrain)
#predicting the output for testing data
ytest["predict_xgboost"]=xgb2.predict(xtest)
#rmse for train data
rmse_train_xgboost=np.sqrt(np.mean((ytrain.price -ytrain.predict_xgboost)**2))
#rmse for testing data
rmse_test_xgboost=np.sqrt(np.mean((ytest.price -ytest.predict_xgboost)**2))
################################################################################################
#building model with support vector machines
svr=SVR(kernel='linear')
svr.fit(xtrain,ytrain.price)

#predicting for train data
ytrain['predict_svm']=svr.predict(xtrain)
#predicting for test data
ytest['predict_svm']=svr.predict(xtest)
#rmse for train data
rmse_train_svm=np.sqrt(np.mean((ytrain.price -ytrain.predict_svm)**2))
#rmse for testing data
rmse_test_svm=np.sqrt(np.mean((ytest.price -ytest.predict_svm)**2))
#################################################################################################
#building model with random forest regressor
rf=RandomForestRegressor(n_estimators=250,n_jobs=3,max_features=40)
rf.fit(xtrain,ytrain.price)
##predicting for train data
ytrain['predict_rf']=rf.predict(xtrain)
#predicting for test data
ytest['predict_rf']=rf.predict(xtest)
#rmse for train data
rmse_train_rf=np.sqrt(np.mean((ytrain.price -ytrain.predict_rf)**2))
#rmse for testing data
rmse_test_rf=np.sqrt(np.mean((ytest.price -ytest.predict_rf)**2))
########################################################################
#linear regression
train=pd.concat([xtrain,ytrain.price],axis=1)
test=pd.concat([xtest,ytest.price],axis=1)

train.columns
#renaming a column name
train.rename(columns={'symboling_-1':'symboling_neg_1','make_mercedes-benz':'make_mercedes_benz'},inplace=True)
linear=smf.ols('price~normalized_losses+ wheel_base+ length+ width+ height+curb_weight+ engine_size+ bore+ stroke+ compression_ratio+horsepower+ peak_rpm+ city_mpg+ highway_mpg+ symboling_neg_1+symboling_0+ symboling_1+ symboling_2+ symboling_3+ make_audi+make_bmw+ make_chevrolet+ make_dodge+ make_honda+ make_isuzu+make_jaguar+ make_mazda+ make_mercedes_benz+ make_mercury+make_mitsubishi+ make_nissan+ make_peugot+ make_plymouth+make_porsche+ make_renault+ make_saab+ make_subaru+make_toyota+ make_volkswagen+ make_volvo+ fuel_type_gas+aspiration_turbo+ num_of_doors_two+ body_style_hardtop+body_style_hatchback+ body_style_sedan+ body_style_wagon+drive_wheels_fwd+ drive_wheels_rwd+ engine_location_rear+engine_type_dohcv+ engine_type_l+ engine_type_ohc+engine_type_ohcf+ engine_type_ohcv+ engine_type_rotor+num_of_cylinders_five+ num_of_cylinders_four+num_of_cylinders_six+ num_of_cylinders_three+num_of_cylinders_twelve+ num_of_cylinders_two+ fuel_system_2bbl+fuel_system_4bbl+ fuel_system_idi+ fuel_system_mfi+fuel_system_mpfi+ fuel_system_spdi+ fuel_system_spfi+ price',data=train).fit()
linear.summary()

