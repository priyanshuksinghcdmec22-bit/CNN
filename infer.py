import argparse, glob, os, cv2, numpy as np
from tensorflow.keras.models import load_model
IMG_SIZE=(128,128)

def run_inference(model_path,input_folder,out_folder='outputs/infer'):
    model=load_model(model_path)
    os.makedirs(out_folder,exist_ok=True)
    for p in glob.glob(input_folder+'/*'):
        img=cv2.imread(p); orig=img.copy()
        img2=cv2.resize(img,IMG_SIZE).astype('float32')/255.0
        pred=model.predict(np.expand_dims(img2,0)); cls=np.argmax(pred)
        label='CRACK' if cls==1 else 'NO_CRACK'
        cv2.putText(orig,label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.imwrite(os.path.join(out_folder,os.path.basename(p)),orig)
    print('Output saved to',out_folder)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model'); p.add_argument('--input_folder')
    a=p.parse_args(); run_inference(a.model,a.input_folder)