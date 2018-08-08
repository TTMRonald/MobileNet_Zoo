import sys, os
sys.path.insert(0, os.path.abspath('..'))
import argparse as ap
import keras
from keras.datasets import cifar10
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from networks.mobilenet_v1 import MobileNet_V1
from networks.mobilenet_v2 import MobileNet_V2

batch_size = 64
num_classes = 10
epochs = 500

if __name__ == "__main__":
	parser = ap.ArgumentParser()
	parser.add_argument("--net", help="network", required="True")
	args = parser.parse_args()

	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	img_input = keras.layers.Input(shape=(32, 32, 3))

	if args.net == 'mobilenet_v1':
		model = MobileNet_V1(input_tensor=img_input, classes=num_classes)
		log_dir = 'logs/mobilenet_v1/000'
	if args.net == 'mobilenet_v2':
		model = MobileNet_V2(input_tensor=img_input, classes=num_classes)
		log_dir = 'logs/mobilenet_v2/000'
	else:
		raise ValueError('You must input the model id with -m')

	model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	print("Training model.")

	logging = TensorBoard(log_dir=log_dir)
	checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
	    monitor='val_loss', save_weights_only=True, save_best_only=True, mode='auto', period=1)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          validation_data=(x_test, y_test),
	          shuffle=True,
	          verbose=1,
	          callbacks=[logging, checkpoint, reduce_lr])
	model.save_weights('logs/trained_weights.h5')
