 # placeholder content from canvas simplified version
import os
from PIL import Image
import numpy as np
IMG_SIZE=(128,128)

def load_dataset(folder):
    pos_dir=os.path.join(folder,'crack')
    neg_dir=os.path.join(folder,'no_crack')
    X=[]; y=[]
    for d,label in [(pos_dir,1),(neg_dir,0)]:
        for p in os.listdir(d):
            if p.lower().endswith(('.jpg','.png','.jpeg')):
                img=Image.open(os.path.join(d,p)).convert('RGB').resize(IMG_SIZE)
                X.append(np.array(img)/255.0); y.append(label)
    return np.array(X,dtype=np.float32), np.array(y,dtype=np.int32)

def simple_train_val_split(X,y,val_ratio=0.2,seed=42):
    import numpy as np
    np.random.seed(seed); idx=np.arange(len(X)); np.random.shuffle(idx)
    s=int(len(X)*(1-val_ratio)); return X[idx[:s]],y[idx[:s]],X[idx[s:]],y[idx[s:]]