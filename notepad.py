import time
import mido

def extract_midi_notes(file_path):
    """Extracts note sequences from a MIDI file."""
    midi_file = mido.MidiFile(file_path)
    note_sequence = []
    data = []
    track = midi_file.tracks[1]
    for msg in track:
        print(msg)
        data.append(msg)
        if msg.type == 'note_on':
            note_sequence.append([msg.note, msg.velocity, msg.time])
    #save data to a text file
    with open("datab.txt", "w") as f:
        for msg in data:
            f.write(str(msg) + "\n") 
    # print (note_sequence[:20])
    return note_sequence

def play_midi(midi_file):
    """Plays a MIDI file."""
    print(mido.get_output_names())
    with mido.open_output('Microsoft GS Wavetable Synth 0') as port:
        for msg in midi_file.play():
            port.send(msg)
            time.sleep(msg.time)

def create_midi_from_notes(note_sequence, output_file):
    """Creates a MIDI file from a note sequence."""
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    tempo = 429523

    track.append(mido.MetaMessage('set_tempo', tempo=429523, time=120))
    for note in note_sequence:
        note_duration = note[2]
        velocity = note[1]
        note = note[0]
        track.append(mido.Message('note_on', note=note, velocity=velocity, time=note_duration))
    midi.save(output_file)

def save_notes(note_seq, output_file):
    """Saves note sequence to a text file."""

    with open(output_file, "w") as f:
        data = ""
        for note in note_seq:
            for value in note:
                data += str(value) + "-"
            data = data[:-1]
            data += ","    
        f.write(data)

def alternate_velocity(pitch_list):
    pitch_occurrences = {}
    result = []

    for pitch in pitch_list:
        if pitch in pitch_occurrences:
            # If it's the second occurrence, set velocity to 0
            result.append(0)
            del pitch_occurrences[pitch]  # Reset the count for this pitch
        else:
            # If it's the first occurrence, set a standard non-zero velocity (e.g., 64)
            result.append(64)
            pitch_occurrences[pitch] = True

    return result

def load_notes(input_file):
    """Loads note sequence from a text file."""
    with open(input_file, "r") as f:
        data = f.read()
        notes = []
        for note in data.split(","):
            note = note.split("-")
            if len(note) == 3:
                notes.append([int(note[0]), int(note[1]), int(note[2])])
        return notes

if __name__ == '__main__':
    midi_path = "midi\schub_d960_3.mid"
    note_sequence = extract_midi_notes(midi_path)
    note_sequence[1] = alternate_velocity(note_sequence[0])
    save_notes(note_sequence, "notes.txt")
    note_sequencea = load_notes("notes.txt")
    print(note_sequencea)
    create_midi_from_notes(note_sequencea, 'output_midi_file.mid')
    play_midi(mido.MidiFile('output_midi_file.mid'))
