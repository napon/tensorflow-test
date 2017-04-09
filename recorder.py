import alsaaudio
import wave 
import datetime
from StringIO import StringIO

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
        
        return inp, FRAME_RATE

def record(inp, duration):
        buffer = []

        for i in range(duration):
                l, data = inp.read()
         
                if l: 
                        buffer.append(data)
	return buffer

def create_soundfile(buffer):
	#n = datetime.datetime.now().strftime('%y%m%d-%H%M%S') + '.wav'
        WRITE_MODE = 'wb'
        CHANNELS = 1 
        SAMPWIDTH = 2
        FRAME_RATE = 44100     
	stream = StringIO()
	f = wave.open(stream, WRITE_MODE)
        f.setnchannels(CHANNELS)
        f.setsampwidth(SAMPWIDTH)
        f.setframerate(FRAME_RATE)
        f.writeframes(''.join(buffer))
        f.close()

	return stream 
