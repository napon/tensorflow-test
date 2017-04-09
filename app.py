#import recorder
import predictor
#from multiprocessing import Pool as ThreadPool
from multiprocessing.dummy import Pool as ThreadPool 

SOUND_CARD = 'hw:1,0'
DURATION_IN_SECONDS = 5
PRETRAINED_MODEL_PATH = 'salamon-cnn-model.h5'
THREAD_POOL_SIZE = 5

def run():
	#inp = recorder.create_input(SOUND_CARD)
	model, graph = predictor.setup_model(PRETRAINED_MODEL_PATH)
	
	files = ['7061-6-0-0.wav', '9031-3-1-0.wav', '14113-4-0-0.wav', '21684-9-0-7.wav', '31323-3-0-2.wav', '40722-8-0-1.wav', '46669-4-0-9.wav', '50901-0-1-0.wav', '69304-9-0-15.wav', '97317-2-0-25.wav', '103074-7-3-1.wav', '175845-1-0-0.wav', '184355-1-0-0.wav']
	soundfiles = []	
	for name in files:
		sound_filename = 'audio/testfold1/' + name
		# Record sound clips of specified duration
		#buffer = recorder.record(inp, DURATION_IN_SECONDS)
		# Create .wav file
		#sound_filename = recorder.create_soundfile(buffer)	
		soundfiles.append(sound_filename)
		if len(soundfiles) % THREAD_POOL_SIZE == 0:
			pool = ThreadPool(THREAD_POOL_SIZE)
			args = []
			for file in soundfiles:
				args.append((graph, model, file))
			pool.map(predictor.predict, args)
			soundfiles = []
		

if __name__ == "__main__":
	run()
