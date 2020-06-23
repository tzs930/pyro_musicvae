import mido
from mido import MidiFile
import numpy as np

def write_midi(pr, ticks_per_beat, write_path, tempo=80):
    def pr_to_list(pr):
        # List event = (pitch, velocity, time)
        data_num, max_len, N = pr.shape
        T = data_num * max_len
        t_last = 0
        pr_tm1 = np.zeros(N)
        list_event = []
        for t in range(T):
            pr_t = pr[t // max_len][t % max_len]
            mask = (pr_t != pr_tm1)
            if (mask).any():
                for n in range(N):
                    if mask[n]:
                        pitch = n + 19
                        velocity = int(pr_t[n])
                        # Time is incremented since last event
                        t_event = t - t_last
                        t_last = t
                        list_event.append((pitch, velocity, t_event))
            pr_tm1 = pr_t
        return list_event
    # Tempo
    microseconds_per_beat = mido.bpm2tempo(tempo)
    # Write a pianoroll in a midi file
    mid = MidiFile()
    mid.ticks_per_beat = ticks_per_beat

    # Each instrument is a track
    for instrument_name, matrix in pr.items():
        # Add a new track with the instrument name to the midi file
        track = mid.add_track(instrument_name)
        # transform the matrix in a list of (pitch, velocity, time)
        events = pr_to_list(matrix)
        # Tempo
        track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))
        # Add the program_change
        try:
            program = program_change_mapping[instrument_name]
        except:
            # Defaul is piano
            # print instrument_name + " not in the program_change mapping"
            # print "Default value is 1 (piano)"
            # print "Check acidano/data_processing/utils/program_change_mapping.py"
            program = 1
        track.append(mido.Message('program_change', program=program))

        # This list is required to shut down
        # notes that are on, intensity modified, then off only 1 time
        # Example :
        # (60,20,0)
        # (60,40,10)
        # (60,0,15)
        notes_on_list = []
        # Write events in the midi file
        for event in events:
            pitch, velocity, time = event
            if velocity == 0:
                # Get the channel
                track.append(mido.Message('note_off', note=pitch, velocity=0, time=time))
                if pitch in notes_on_list:
                    notes_on_list.remove(pitch)
            else:
                if pitch in notes_on_list:
                    track.append(mido.Message('note_off', note=pitch, velocity=0, time=time))
                    if pitch in notes_on_list:    
                        notes_on_list.remove(pitch)
                    time = 0
                track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=time))
                notes_on_list.append(pitch)
                
    mid.save(write_path)
    return

def np_to_midi(name, tpb=4, tmp=80):
    prs = np.load(name + '.npy', allow_pickle=True)
    
    prs = np.array(np.round(prs, decimals=0), dtype=int)
    prs = np.squeeze(prs)
    prs = (prs / prs.max()) * 97
    print(prs)
    prs = np.array(prs // (128//16) * (128//16))
    prs = prs[:,:,:88]
    pr = np.vstack(prs)
    # pr = prs[0]
    # pr : [sequence_len, pitch_dim] :0~127
    # pr = np.vstack(prs[:,:5])

    pr_dict = {"piano" : np.squeeze(pr)}

    write_midi(pr_dict, tpb, name + ".mid", tempo=tmp)

load_path = 'outputs/Mozart_64_IWAE_num_particle_8/generated_samples'
# load_path = 'datasets/beethoven64_test.npy'
np_to_midi(load_path)