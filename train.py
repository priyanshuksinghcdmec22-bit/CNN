import argparse, os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from simple_cnn import build_simple_cnn
from simple_vit import build_tiny_vit
from data import load_dataset, simple_train_val_split

def main(train_dir, model_type='cnn', epochs=10, out_dir='outputs'):
    X,y=load_dataset(train_dir)
    X_train,y_train,X_val,y_val=simple_train_val_split(X,y)
    model=build_simple_cnn(input_shape=X_train.shape[1:]) if model_type=='cnn' else build_tiny_vit(input_shape=X_train.shape[1:])
    model.compile(optimizer=Adam(1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    os.makedirs(out_dir,exist_ok=True)
    chk=ModelCheckpoint(os.path.join(out_dir,'best_model.h5'),save_best_only=True,monitor='val_accuracy')
    model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=epochs,batch_size=16,callbacks=[chk])
    print('Model saved to',os.path.join(out_dir,'best_model.h5'))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--train_dir'); p.add_argument('--model',default='cnn'); p.add_argument('--epochs',type=int,default=10)
    a=p.parse_args(); main(a.train_dir,a.model,a.epochs)