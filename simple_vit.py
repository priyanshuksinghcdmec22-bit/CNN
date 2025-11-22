import tensorflow as tf
from tensorflow.keras import layers, Model

def build_tiny_vit(input_shape=(128,128,3), patch_size=16, projection_dim=32, num_heads=2, mlp_dim=64, num_layers=4, num_classes=2):
    inputs=layers.Input(shape=input_shape)
    patches=tf.image.extract_patches(images=inputs,sizes=[1,patch_size,patch_size,1],strides=[1,patch_size,patch_size,1],rates=[1,1,1,1],padding='VALID')
    patch_dims=patches.shape[-1]
    patches=tf.reshape(patches,[-1,(input_shape[0]//patch_size)*(input_shape[1]//patch_size),patch_dims])
    x=layers.Dense(projection_dim)(patches)
    positions=tf.range(start=0,limit=tf.shape(x)[1],delta=1)
    pos_emb=layers.Embedding(input_dim=1000,output_dim=projection_dim)(positions)
    x=x+pos_emb
    for _ in range(num_layers):
        x1=layers.LayerNormalization()(x)
        attn=layers.MultiHeadAttention(num_heads=num_heads,key_dim=projection_dim)(x1,x1)
        x=layers.Add()([x,attn])
        x2=layers.LayerNormalization()(x)
        mlp=layers.Dense(mlp_dim,activation='relu')(x2)
        mlp=layers.Dense(projection_dim)(mlp)
        x=layers.Add()([x,mlp])
    x=layers.GlobalAveragePooling1D()(x)
    x=layers.Dense(64,activation='relu')(x)
    outputs=layers.Dense(num_classes,activation='softmax')(x)
    return Model(inputs,outputs)