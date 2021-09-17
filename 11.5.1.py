import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers,losses

batches=128
total_words=10000
max_review_len=80
embedding_len=100

(x_train,y_train),(x_test,y_test)=keras.datasets.imdb.load_data(num_words=total_words)
print(x_train.shape,len(x_train[0]),y_train.shape)
print(x_test.shape,len(x_test[0]),y_test.shape)

def get_code():
    word_index=keras.datasets.imdb.get_word_index()
    # for k,v in word_index.items():
    #     print(k,v)
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"]=0
    word_index["<START>"]=1
    word_index["<UNK>"]=2
    word_index["<UNUSED>"]=3
    reverse_word_index=dict([(v,k) for k,v in word_index.items()])
    return reverse_word_index
def decode_review(text):
    reverse_word_index=get_code()
    return ' '.join([reverse_word_index.get(i,'?') for i in text])
# print(decode_review(x_train[0]))


x_train=keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_review_len)
x_test=keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_review_len)
db_train=tf.data.Dataset.from_tensor_slices((x_train,y_train))
db_train=db_train.shuffle(1000).batch(batch_size=batches,drop_remainder=True)
db_test=tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test=db_test.batch(batches,drop_remainder=True)



class MyRNN(keras.Model):
    def __init__(self,units):
        super(MyRNN,self).__init__()
        self.state0=[tf.zeros([batches,units])]
        self.state1=[tf.zeros([batches,units])]
        self.embedding=layers.Embedding(total_words,embedding_len,input_length=max_review_len)
        self.rnn_cell0 = layers.SimpleRNNCell(units,dropout=0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.5)
        self.outlayer=keras.models.Sequential([layers.Dense(1)])
        # self.outlayer = keras.models.Sequential([
        #     layers.Dense(units),
        #     layers.Dropout(rate=0.5),
        #     layers.ReLU(),
        #     layers.Dense(1)])
    def call(self,inputs,training=None):
        x=inputs
        x=self.embedding(x)
        state0=self.state0
        state1=self.state1
        for word in tf.unstack(x,axis=1):
            out0,state0=self.rnn_cell0(word,state0,training)
            out1,state1=self.rnn_cell1(out0,state1,training)
        x=self.outlayer(out1,training)
        prob=tf.sigmoid(x)
        return prob
def main():
    units=64
    epochs=20

    model=MyRNN(units)
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss=losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    # db_train,db_test=preprocess_data(x_train,x_test)

    model.fit(db_train,epochs=epochs,validation_data=db_test)
    # model.fit(x_train,y_train, epochs=epochs, validation_data=db_test)
    model.evaluate(db_test)
main()

