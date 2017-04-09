import alsaaudio
import wave

card = 'hw:1,0'
inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK, card)
inp.setchannels(1)
inp.setrate(44100)
inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)

inp.setperiodsize(160) 

i = 0
buffer = []
seconds = 15

while i <= seconds * 44100: 
	l, data = inp.read()
	i += 1
	 
	if l: 
		buffer.append(data)

n = 'test.wav'
f = wave.open(n, 'wb')
f.setnchannels(1)
f.setsampwidth(2)
f.setframerate(16000)
f.writeframes(''.join(buffer))
f.close()


