import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Concatenate, Input
from tensorflow.keras.models import Model
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, average_precision_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the training and testing data into Pandas DataFrames
train_data = pd.read_csv('C:\\Users\\SBD2RP\\OneDrive - MillerKnoll\\installs\\Desktop\\output\\modified_train_data.csv')
test_data = pd.read_csv('C:\\Users\\SBD2RP\\OneDrive - MillerKnoll\\installs\\Desktop\\output\\modified_test_data.csv')

# Convert data types
train_data['song.sections_start'] = train_data['song.sections_start'].apply(
    lambda x: np.array(x.split(',')).astype(float) if isinstance(x, str) else np.array([])
)
test_data['song.sections_start'] = test_data['song.sections_start'].apply(
    lambda x: np.array(x.split(',')).astype(float) if isinstance(x, str) else np.array([])
)
train_data['input'] = train_data['input'].apply(
    lambda x: float(x) if isinstance(x, str) else np.nan
)
test_data['input'] = test_data['input'].apply(
    lambda x: float(x) if isinstance(x, str) else np.nan
)

# Filter out rows with missing values
train_data = train_data.dropna(subset=['song.sections_start', 'input', 'label'])
test_data = test_data.dropna(subset=['song.sections_start', 'input', 'label'])

# Convert the lists to arrays
X_train_sections = train_data['song.sections_start'].to_list()
X_train_input = np.array(train_data['input'], dtype='float32')
y_train = np.array(train_data['label'], dtype='float32')

X_test_sections = test_data['song.sections_start'].to_list()
X_test_input = np.array(test_data['input'], dtype='float32')
y_test = np.array(test_data['label'], dtype='float32')

# Check if X_train_sections is not empty before applying pad_sequences
if X_train_sections:
    # Pad sequences
    X_train_sections = pad_sequences(X_train_sections, padding='post', truncating='post', dtype='float32')
    X_test_sections = pad_sequences(X_test_sections, maxlen=X_train_sections.shape[1], padding='post', truncating='post', dtype='float32')

    # Reshape X_train_sections and X_test_sections
    X_train_sections = np.expand_dims(X_train_sections, axis=-1)
    X_test_sections = np.expand_dims(X_test_sections, axis=-1)

    # Define the model architecture
    input_sections = Input(shape=(X_train_sections.shape[1], X_train_sections.shape[2]), name='input_sections')
    input_input = Input(shape=(1,), name='input_input')

    sections_lstm = LSTM(units=128)(input_sections)
    input_lstm = LSTM(units=128)(input_input)

    concatenated = Concatenate()([sections_lstm, input_lstm])

    output = Dense(1, activation='sigmoid')(concatenated)

    model = Model(inputs=[input_sections, input_input], outputs=output)

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(x=[X_train_sections, X_train_input], y=y_train, epochs=10, batch_size=32)

    # Evaluate the model
    loss, accuracy = model.evaluate([X_test_sections, X_test_input], y_test)
    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)

    # Make predictions
    predictions = model.predict([X_test_sections, X_test_input])

    # Threshold the predictions
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)

    # Compute evaluation metrics
    precision = precision_score(y_test, binary_predictions)
    recall = recall_score(y_test, binary_predictions)
    accuracy = accuracy_score(y_test, binary_predictions)
    f1 = f1_score(y_test, binary_predictions)
    average_precision = average_precision_score(y_test, predictions)

    print('Precision:', precision)
    print('Recall:', recall)
    print('Accuracy:', accuracy)
    print('F1 Score:', f1)
    print('Average Precision:', average_precision)

    # Save the trained model
    # model.save('C:\\Users\\SBD2RP\\OneDrive - MillerKnoll\\installs\\Desktop\\output\\model.h5')
else:
    print("X_train_sections is empty. Check the data and filtering process.")
