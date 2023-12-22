# hyperparameters
max_length = 50 # max length of tweets
batch_size = 132

# Bert Tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

#Spliiting the data
train_df, test = train_test_split(df, test_size=0.2, random_state=42)
x_train, dev = train_test_split(train_df, test_size=0.10, random_state=42)

print(x_train.shape)
print(test.shape)
print(dev.shape)

train = x_train

labels = train.target.unique().tolist()
labels

encoder = LabelEncoder()
encoder.fit(train.target.tolist())

y_train = encoder.transform(train.target.tolist())
y_test = encoder.transform(test.target.tolist())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("y_train",y_train.shape)
print("y_test",y_test.shape)

def bert_encode(data):
    tokens = tokenizer.batch_encode_plus(data, max_length=max_length, padding='max_length', truncation=True)
    
    return tf.constant(tokens['input_ids'])


train_encoded = bert_encode(train.text_clean)
dev_encoded = bert_encode(dev.text_clean)


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_encoded, y_train))
    .shuffle(32)
    .batch(batch_size)
)

dev_dataset = (
    tf.data.Dataset
    .from_tensor_slices((dev_encoded, y_dev))
    .shuffle(32)
    .batch(batch_size)
)
#Self Attention-Layer
class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def call(self, query, key, value, mask=None):
        d_model = tf.cast(tf.shape(query)[-1], tf.float32)
        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(d_model)
        
        if mask is not None:
            scores += (mask * -1e9)  # Adding a large negative value to masked positions
        
        attention_weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

def SatCoBiLSTM_model():
    bert_encoder = TFBertModel.from_pretrained(model_name)
    input_word_ids = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    last_hidden_states = bert_encoder(input_word_ids)[0]
    x = tf.keras.layers.SpatialDropout1D(0.3)(last_hidden_states)
    #convolution layer of different filter sizes
    conv_layers = []
    for filter_size in [3, 4, 5]:
        conv_layer = tf.keras.layers.Conv1D(128, filter_size, activation='relu')(x)
        # Apply padding to make all sequences of equal length before concatenation
        conv_layer = tf.keras.layers.ZeroPadding1D(padding=(0, filter_size - 1))(conv_layer)
        conv_layers.append(conv_layer)
    
    # Use GlobalMaxPooling to unify the sequences length
    pooled_layers = [tf.keras.layers.MaxPooling1D(pool_size=max_length - filter_size + 1)(conv_layer)
                     for conv_layer, filter_size in zip(conv_layers, [3, 4, 5])]

    if len(pooled_layers) > 1:
        x = tf.keras.layers.Concatenate(axis=-1)(pooled_layers)
    else:
        x = pooled_layers[0]


    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.3, recurrent_dropout=0.2, return_sequences=True))(x)
    
    attention_output, _ = ScaledDotProductAttention()(x, x, x)
    x = tf.keras.layers.Dense(128, activation='relu')(attention_output)
    
    outputs = tf.keras.layers.Dense(27, activation='softmax')(x)
    model = tf.keras.Model(input_word_ids, outputs)
    
    return model


with strategy.scope():
    model = SatCoBiLSTM_model()
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=adam_optimizer,metrics=['accuracy'])

    model.summary()

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=3, cooldown=0),
              EarlyStopping(monitor='val_loss', patience=3)]

history = model.fit(
    train_dataset,
    batch_size=batch_size,
    epochs=20,
    validation_data=dev_dataset,
    verbose=1,
    callbacks = callbacks)

test_encoded = bert_encode(test.text_clean)

#predicted tweets on test data
test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_encoded)
    .batch(batch_size)
)

y_pred = []
predicted_tweets = model.predict(test_dataset, batch_size=batch_size)
predicted_tweets_binary = tf.cast(tf.round(predicted_tweets), tf.int32).numpy().flatten()

%%time
scores = model.evaluate(test_encoded, y_test, batch_size=batch_size)
print()
print("ACCURACY:",scores[1])
print("LOSS:",scores[0])

#Classification report
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)

cnf_matrix = confusion_matrix(y_test, predicted_tweets_binary)
plt.figure(figsize=(6,6))
plot_confusion_matrix(cnf_matrix, classes=train.target.unique(), title="Confusion matrix")
plt.show()

print('Precision: %.4f' % precision_score(y_test, predicted_tweets_binary))
print('Recall: %.4f' % recall_score(y_test, predicted_tweets_binary))
print('Accuracy: %.4f' % accuracy_score(y_test, predicted_tweets_binary))
print('F1 Score: %.4f' % f1_score(y_test, predicted_tweets_binary))
print(classification_report(y_test, predicted_tweets_binary))