import alsaaudio
import wave 
import datetime

def create_input(sound_card):
        inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK, sound_card)
        CHANNELS = 1
        FRAME_RATE = 44100
        FORMAT = alsaaudio.PCM_FORMAT_S16_LE
        PERIOD = 160

        inp.setchannels(CHANNELS)
        inp.setrate(FRAME_RATE)
        inp.setformat(FORMAT)
        inp.setperiodsize(PERIOD) 
        
        return inp

def record(inp, duration):
        buffer = []

        for i in range(duration**FRAME_RATE):
                l, data = inp.read()
         
                if l: 
                        buffer.append(data)

def create_soundfile(buffer):
        now = datetime.datetime.now()
        timestamp_str = now.strftime('%y%m%d-%H%M%S')
        name = timestamp_str + '.wav'
        WRITE_MODE = 'wb'
        CHANNELS = 1
        SAMPWIDTH = 2
        FRAME_RATE = 16000      
        f = wave.open(name, WRITE_MODE)
        f.setnchannels(CHANNELS)
        f.setsampwidth(SAMPWIDTH)
        f.setframerate(FRAME_RATE)
        f.writeframes(''.join(buffer))
        f.close()

        return name     

