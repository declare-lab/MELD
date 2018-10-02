from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, Lambda, LSTM, TimeDistributed, Masking, Bidirectional, GRU
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Model, load_model
import keras.backend as K
from sklearn.model_selection import train_test_split
from data_helpers import load_bimodal_data
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from keras import optimizers
import os, pickle
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


print('Loading data')
train_x, val_x, test_x, train_y, val_y, test_y, train_mask, val_mask, test_mask, train_id, val_id, test_id, MODE = load_bimodal_data()

TRAIN = True
print("Model is in {} mode and Training is: {}".format(MODE, str(TRAIN)))

PATH = "./models/bimodal_{}.hdf5".format(MODE.lower())
sequence_length = train_x.shape[1]
embedding_dim = train_x.shape[2]
classes = train_y.shape[2]
drop = 0.5

epochs = 400
batch_size = 100

def calc_test_result(result, test_label, test_mask):

	true_label=[]
	predicted_label=[]

	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			if test_mask[i,j]==1:
				true_label.append(np.argmax(test_label[i,j] ))
				predicted_label.append(np.argmax(result[i,j] ))

	# for i in range(result.shape[0]):
	# 	true_label.append(np.argmax(test_label[i] ))
	# 	predicted_label.append(np.argmax(result[i] ))
		
	print("Confusion Matrix :")
	print(confusion_matrix(true_label, predicted_label))
	print("Classification Report :")
	print(classification_report(true_label, predicted_label,digits=4))
	print("Accuracy ", accuracy_score(true_label, predicted_label))

	test_cmat = confusion_matrix(true_label, predicted_label)
	print('Classwise accuracies: \n ', test_cmat.diagonal()*1.0/test_cmat.sum(axis=1))
	print('Micro FScore: \n ', precision_recall_fscore_support(true_label, predicted_label, average='micro'))
	print('Weighted FScore: \n ', precision_recall_fscore_support(true_label, predicted_label, average='weighted'))

if TRAIN:

	print("Creating Model...")
	
	inputs = Input(shape=(sequence_length, embedding_dim), dtype='float32')
	masked = Masking(mask_value=0)(inputs)
	# lstm = Bidirectional(LSTM(600, activation='relu', return_sequences = True, dropout=0.3))(masked)
	lstm = Bidirectional(GRU(100, activation='tanh', return_sequences = True), name="utter")(masked)
	#lstm = (GRU(300, activation='tanh', return_sequences = True,name="utter"))(masked)

	inter = Dropout(0.1)(lstm)
	#inter1 = TimeDistributed(Dense(500,activation='relu'))(inter)
	#inter = Dropout(0.5)(inter1)
	#inter1 = TimeDistributed(Dense(512,activation='tanh'))(inter1)
	#inter = Dropout(0.5)(inter1)
	output = TimeDistributed(Dense(classes,activation='softmax'))(inter)

	model = Model(inputs, output)
	# aux = Model(inputs, cnn_outputs)
	checkpoint = ModelCheckpoint(PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
	opti = optimizers.Adadelta(lr=0.01, rho=0.85, epsilon=None, decay=0.0)
	#opti = optimizers.Adam(lr=0.0001)
	model.compile(optimizer=opti, loss='categorical_crossentropy', sample_weight_mode='temporal')

	early_stopping = EarlyStopping(monitor='val_loss', patience=10)
	model.fit(train_x, train_y,
	                epochs=epochs,
	                batch_size=batch_size,
	                sample_weight=train_mask,
	                shuffle=True, 
	                callbacks=[early_stopping, checkpoint],
	                validation_data=(val_x, val_y, val_mask))

	# model.save('./models/'+mode+'.h5') 

	# result_test = model.predict(test_x)
	# result_val=model.predict(val_x)
	# result_train = model.predict(train_x)

	# calc_test_result(result_val, val_y, val_mask)	
	# print()
	# calc_test_result(result_test, test_y, test_mask)


	# Final test
	model = load_model(PATH)
	intermediate_layer_model = Model(input=model.input, output=model.get_layer("utter").output)

	intermediate_output_train = intermediate_layer_model.predict(train_x)
	intermediate_output_val = intermediate_layer_model.predict(val_x)
	intermediate_output_test = intermediate_layer_model.predict(test_x)

	print(model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1, sample_weight=None))


	calc_test_result(model.predict(test_x), test_y, test_mask)

else:

	model = load_model(PATH)
	intermediate_layer_model = Model(input=model.input, output=model.get_layer("utter").output)

	intermediate_output_train = intermediate_layer_model.predict(train_x)
	intermediate_output_val = intermediate_layer_model.predict(val_x)
	intermediate_output_test = intermediate_layer_model.predict(test_x)

	print(model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1, sample_weight=None))


	calc_test_result(model.predict(test_x), test_y, test_mask)

	train_emb, val_emb, test_emb = {}, {}, {}
	for idx, ID in enumerate(train_id):
	    train_emb[ID] = intermediate_output_train[idx]
	for idx, ID in enumerate(val_id):
	    val_emb[ID] = intermediate_output_val[idx]
	for idx, ID in enumerate(test_id):
	    test_emb[ID] = intermediate_output_test[idx]

	pickle.dump([train_emb, val_emb, test_emb], open("./output/bimodal_{}.pkl".format(MODE.lower()), "wb"))


