import numpy as np
np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(0)
from keras.models import Sequential
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from keras.layers import Conv1D,MaxPooling1D,Dropout,Dense,Flatten
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score


def main():
	# which Convolutional architecture to choose from the define architectures function
	arch_number = 3

	#which dataset to use
	filename = 'concatenated_100.csv'

	#read in the csv file and prepare data for classification
	x_train,x_test,y_train,y_test = read_and_preprocess_data(filename)

	#window length and input shape
	window_length = len(x_train[0])/2
	input_shape = (x_train.shape[1],1)

	
	architectures = define_architectures(window_length,input_shape)
	chosen_architecture = architectures[arch_number]
	results = classification(chosen_architecture,x_train,y_train)

def classification(model,x_train,y_train):
	#create the stratified k fold index maker
	kfold = StratifiedKFold(n_splits=3,random_state=0)
	#compile model and choose its loss function and optimizer. 'metrics' allows you to add more metrics reported per epoch
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	f_scores = []
	accuracies = []
	converted_labels = [ 1 if x == [0,1] else 0 for x in y_train]

	#for all train/test splits in the k fold cross validation:
	for train_index, test_index in kfold.split(x_train, converted_labels):
		x = x_train[train_index]
		y = [y_train[index] for index in train_index]
		#train on the training set
		model.fit(x,y,batch_size = 8,verbose=1,epochs=1)

		#predict on the validation set
		predictions = model.predict(x_train[test_index])

		# get all summary statistics for the fold
		y_to_check = [converted_labels[index] for index in test_index]
		converted_predictions = [np.argmax(prediction) for prediction in predictions]
		print converted_predictions
		print predictions
		f_score = f1_score(y_to_check,converted_predictions)
		accuracy = accuracy_score(y_to_check,converted_predictions)
		f_scores.append(f_score)
		accuracies.append(accuracy)
	print "F-Scores on each split: "
	print f_scores
	print "Mean F score"
	print np.mean(f_scores)

	print "Accuracies on each split: "
	print accuracies
	print "Mean Accuracy: "
	print np.mean(accuracies)

def define_architectures(window_length,input_shape):
	#arbitrarily (Sort of ) defined conv architecutres
	model1 = Sequential()
	model1.add(Conv1D(input_shape = input_shape,filters=16,kernel_size=window_length))
	model1.add(Dropout(.2))
	model1.add(Flatten())
	model1.add(Dense(2,activation = 'sigmoid'))

	model2 = Sequential()
	model2.add(Conv1D(input_shape = input_shape,filters=16,kernel_size=window_length,activation='relu'))
	model2.add(Dropout(.1))
	model2.add(Conv1D(filters=8,kernel_size=window_length/2))
	model2.add(Flatten())
	model2.add(Dense(2,activation = 'sigmoid'))

	model3 = Sequential()
	model3.add(Conv1D(input_shape = input_shape,filters=16,kernel_size=window_length))
	model3.add(Dropout(.2))
	model3.add(Conv1D(filters=8,kernel_size=window_length/2))
	model3.add(MaxPooling1D(pool_size=2))
	model3.add(Dropout(.3))
	model3.add(Flatten())
	model3.add(Dense(2,activation = 'sigmoid'))


	model4 = Sequential()
	model4.add(Conv1D(input_shape = input_shape,filters=32,kernel_size=window_length))
	model4.add(Dropout(.2))
	model4.add(Conv1D(filters=16,kernel_size=window_length/2))
	model4.add(MaxPooling1D(pool_size=2))
	model4.add(Dropout(.3))
	model4.add(Flatten())
	model4.add(Dense(8,activation='relu'))
	model4.add(Dense(2,activation = 'sigmoid'))

	architectures = [model1,model2,model3,model4]
	return architectures

#sanity check with given answers

#kullback_leibler_divergence

def read_and_preprocess_data(filename):
	data = pd.read_csv(filename)

	#### oversample 1s
	data = pd.concat([data,data[data['Labels'] == 1]])
	#grab the labels from 
	labels = data['Labels']
	new_labels = []
	print 'Number of 1\'s: '+str(len([x for x in labels if x == 1]))
	print 'Number of 0\'s: '+str(len([x for x in labels if x == 0]))
	print 'total length: '+str(len(labels))
	features = []
	#split each nucleotide string into a vector of nucleotides
	for sequence in data['0'].values:
		features.append([nucleotide for nucleotide in sequence])
	#create the label encoder that assigns each nucleotide its own unique integer
	le = LabelEncoder()
	le.fit(features[0]+['n'])
	new_labels = []
	#convert our labels to vectors
	for i in labels:
		if i == 1:
			new_labels.append([0,1])
		else:
			new_labels.append([1,0])
	#use the label encoder
	features_transformed = [le.transform(sample) for sample in features]
	#split the dataset into a stratified training and test split (20% of data is held out here)
	x_train,x_test,y_train,y_test = train_test_split(features_transformed,new_labels,test_size = .2,random_state =0 )
	#reshape training set so it fits in the model
	x_train = np.expand_dims(np.array(x_train), axis=2)
	x_test = np.expand_dims(np.array(x_test), axis=2)
	return x_train,x_test,y_train,y_test

main()