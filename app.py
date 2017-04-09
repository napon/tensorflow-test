import recorder
from multiprocessing.dummy import Pool as ThreadPool 
import requests
import wave
from StringIO import StringIO

SOUND_CARD = 'hw:1,0'
DURATION_IN_SECONDS = 10 
THREAD_POOL_SIZE = 2 
SERVER_URL = 'http://172.25.96.198:5000'

def run():
	inp, frame_rate = recorder.create_input(SOUND_CARD)
	buffers = []
	
	while 1:	
		# Record sound clips of specified duration
		buffer = recorder.record(inp, DURATION_IN_SECONDS * frame_rate)
		buffers.append(buffer)
		if len(buffers) % THREAD_POOL_SIZE == 0:
			pool = ThreadPool(THREAD_POOL_SIZE)
			pool.map(send_buffer, buffers)
			soundfiles = []
		

def send_buffer(buffer):
	CHANNELS = 1
	SAMPWIDTH = 2
	FRAME_RATE = 44100
	WRITE_MODE = 'wb'	
	stream = StringIO()
	f = wave.open(stream, WRITE_MODE)
	f.setnchannels(CHANNELS)
	f.setsampwidth(SAMPWIDTH)
	f.setframerate(FRAME_RATE)
	f.writeframes(''.join(buffer))
	f.close()
	
	requests.post(url=SERVER_URL, files={'file': stream})


if __name__ == "__main__":
	run()
