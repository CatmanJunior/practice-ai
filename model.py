# a tensorflow LSTM for generating midi files
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, Input, LeakyReLU
import numpy as np

from notepad import load_notes,  create_midi_from_notes, alternate_velocity


import numpy as np



def generate_sequence(model, seed_sequence, num_notes_to_generate):
    generated_sequence = []

    for _ in range(num_notes_to_generate):
        # Predict the next note (one-hot encoded)
        prediction = model.predict(seed_sequence, verbose=0)[0]
        
        # Convert the prediction to an actual note
        predicted_note = np.argmax(prediction)
        

        # Create the next input sequence
        next_sequence = np.zeros((1, seed_sequence.shape[1], seed_sequence.shape[2]))
        next_sequence[0, :-1, :] = seed_sequence[0, 1:, :]
        next_sequence[0, -1, predicted_note] = 1

        seed_sequence = next_sequence
        int_to_note = dict((number, note) for number, note in enumerate(set(notes)))
        predicted_note = int_to_note[predicted_note]
        generated_sequence.append(predicted_note)
    return generated_sequence


def normalize_input(input_data, max_value, min_value):
    # normalize input from min to max, to 0 to 1
    return (input_data - min_value) / (max_value - min_value)



def CreateTrainData(notes, sequence_length):
    # Prepare sequences and corresponding labels
    x_train = []
    y_train = []
    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        x_train.append(sequence_in)
        y_train.append(sequence_out)

    # Convert the sequences into a numpy array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train

def EncodeTrainData(notes, sequence_length, x_train, y_train):
    notes_to_int = dict((note, number) for number, note in enumerate(set(notes)))
    x_train_int = []
    y_train_int = []
    for i in range(x_train.shape[0]):
        x_train_int.append([notes_to_int[note] for note in x_train[i]])
        y_train_int.append(notes_to_int[y_train[i]])
    
    # Convert the encoded sequences into a numpy array
    x_train_encoded = np.zeros((len(x_train_int), sequence_length, len(set(notes))))
    y_train_encoded = np.zeros((len(y_train_int), len(set(notes))))
    for i, sequence in enumerate(x_train_encoded):
        for j, note in enumerate(sequence):
            x_train_encoded[i, j, x_train_int[i][j]] = 1
        y_train_encoded[i, y_train_int[i]] = 1
    return x_train_encoded, y_train_encoded




midi_notes = load_notes("notes.txt")
#get rid of the velocity and time
notes = [note[0] for note in midi_notes]
time = [note[2] for note in midi_notes]
velocity = [note[1] for note in midi_notes]

n_unique_notes = len(set(notes))

print("Number of notes:", len(notes))
print("Unique notes:", n_unique_notes)

frequency_count = Counter(notes)

# frequency_count is now a dictionary where each key is a unique integer from int_list
# and the corresponding value is the frequency of that integer in int_list
for unique_int, count in frequency_count.items():
    print(f"The number {unique_int} appears {count} times.")

sequence_length = 25  # Length of each input sequence
unique_features = n_unique_notes

# define the model
model = Sequential([
    Input(shape=(sequence_length, unique_features)),
    LSTM(256, return_sequences=True),
    LSTM(128),
    Dense(256),
    LeakyReLU(alpha=0.01),
    Dense(unique_features),
    Activation('softmax')
])

x_train, y_train = CreateTrainData(notes, sequence_length)
print (notes[:10])
print(x_train[:4])
print(y_train[:7])
x_train_encoded, y_train_encoded = EncodeTrainData(notes, sequence_length, x_train, y_train)
print(x_train_encoded[:2])

print(y_train_encoded[:7])

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x_train_encoded,y_train_encoded , epochs=1, batch_size=16)
model.fit(x_train_encoded,y_train_encoded , epochs=1, batch_size=16)

# Example usage
seed_sequence = x_train_encoded[0:1] # Using the first sequence of the training data as the seed
num_notes_to_generate = 200
new_sequence = generate_sequence(model, seed_sequence, num_notes_to_generate)
print(new_sequence) 
frequency_count = Counter(new_sequence)

# frequency_count is now a dictionary where each key is a unique integer from int_list
# and the corresponding value is the frequency of that integer in int_list
for unique_int, count in frequency_count.items():
    print(f"The number {unique_int} appears {count} times.")

new_velocity = alternate_velocity(new_sequence)

#combine the notes with the time and velocity
combined_notes = []
for i in range(len(new_sequence)):
    combined_notes.append([new_sequence[i], new_velocity[i], time[i]])
print(combined_notes[:10])


create_midi_from_notes(combined_notes, "generated.mid")