# coding:utf-8
import pyaudio
import wave
import json
import signal
import sys
import os

RECORD_RATE = 16000
RECORD_CHANNELS_DEFAULT = 1
RECORD_CHANNELS = 4
RECORD_WIDTH = 2
CHUNK = 1024
RECORD_SECONDS = 5

OUTPUT_ROOT = "../../data/CB313"
RECORD_COORDINATES = "_3_3"

WAVE_OUTPUT_FILENAME = OUTPUT_ROOT + "/output" + RECORD_COORDINATES + ".wav"
WAVE_OUTPUT_FILENAME1 = OUTPUT_ROOT + "/output1.wav"
WAVE_OUTPUT_FILENAME2 = OUTPUT_ROOT + "/output2.wav"
WAVE_OUTPUT_FILENAME3 = OUTPUT_ROOT + "/output3.wav"
WAVE_OUTPUT_FILENAME4 = OUTPUT_ROOT + "/output4.wav"
# RECORD_DEVICE_NAME = "seeed-2mic-voicecard"
RECORD_DEVICE_NAME = "USB Camera-B4.09.24.1"

p = pyaudio.PyAudio()
stream = p.open(
	rate=RECORD_RATE,
	format=p.get_format_from_width(RECORD_WIDTH),
	channels=RECORD_CHANNELS_DEFAULT,
	input=True,
	start=False)

wave_file = wave.open(WAVE_OUTPUT_FILENAME, "wb")
#wave_file1 = wave.open(WAVE_OUTPUT_FILENAME1, "wb")
#wave_file2 = wave.open(WAVE_OUTPUT_FILENAME2, "wb")
#wave_file3 = wave.open(WAVE_OUTPUT_FILENAME3, "wb")
#wave_file4 = wave.open(WAVE_OUTPUT_FILENAME4, "wb")

buffer1 = list(range(CHUNK))
buffer2 = list(range(CHUNK))
buffer3 = list(range(CHUNK))
buffer4 = list(range(CHUNK))

def open_files():
	wave_file.setnchannels(RECORD_CHANNELS)
	wave_file.setsampwidth(2)
	wave_file.setframerate(RECORD_RATE)

	# wave_file1.setnchannels(RECORD_CHANNELS)
	# wave_file1.setsampwidth(2)
	# wave_file1.setframerate(RECORD_RATE)
	#
	# wave_file2.setnchannels(RECORD_CHANNELS)
	# wave_file2.setsampwidth(2)
	# wave_file2.setframerate(RECORD_RATE)
	#
	# wave_file3.setnchannels(RECORD_CHANNELS)
	# wave_file3.setsampwidth(2)
	# wave_file3.setframerate(RECORD_RATE)
	#
	# wave_file4.setnchannels(RECORD_CHANNELS)
	# wave_file4.setsampwidth(2)
	# wave_file4.setframerate(RECORD_RATE)

def close_files():
	wave_file.close()
	# wave_file1.close()
	# wave_file2.close()
	# wave_file3.close()
	# wave_file4.close()

def record():
	open_files()

	stream.start_stream()
	print("* recording")


	for i in range(0, int(RECORD_RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		#print("length of data: %d" %(len(data)))

		for j in range(CHUNK):
			#assert((data[j*8] | (data[j*8 + 1] << 8)) == data[j*8]+data[j*8+1]*256)
			#print("%x" %(data[j*8] | (data[j*8 + 1] << 8)),
			#	  "\t%x %x" %(data[j*8 + 2], data[j*8 + 3]),
			#	  "\t%x %x" % (data[j*8 + 4], data[j*8 + 5]),
			#	  "\t%x %x" % (data[j*8 + 6], data[j*8 + 7])
			#	  )

			#bytes_buffer1 = bytes_buffer1 + data[j*8 + 0]
			#bytes_buffer1[j*2 + 1] = data[j*8 + 1]
			#bytes_buffer1[j*2 + 0] = data[j*8 + 2]
			#bytes_buffer1[j*2 + 1] = data[j*8 + 3]
			#bytes_buffer1[j*2 + 0] = data[j*8 + 4]
			#bytes_buffer1[j*2 + 1] = data[j*8 + 5]
			#bytes_buffer1[j*2 + 0] = data[j*8 + 6]
			#bytes_buffer1[j*2 + 1] = data[j*8 + 7]

			buffer1[j] = data[j*8 + 0] | (data[j*8 + 1] << 8)
			buffer2[j] = data[j*8 + 2] | (data[j*8 + 3] << 8)
			buffer3[j] = data[j*8 + 4] | (data[j*8 + 5] << 8)
			buffer4[j] = data[j*8 + 6] | (data[j*8 + 7] << 8)
			if j == 0 and i == 0:
				print("%x\t%x\t%x\t%x" %(buffer1[j], buffer2[j], buffer3[j], buffer4[j]))


		wave_file.writeframes(data)
		#wave_file1.writeframes(bytes_buffer1)
		#wave_file2.writeframes(bytes_buffer2)
		#wave_file3.writeframes(bytes_buffer3)
		#wave_file4.writeframes(bytes_buffer4)

	print("* done recording")
	stream.stop_stream()
	close_files()
	# audio_data should be raw_data
	return ("record end")


def sigint_handler(signum, frame):
	stream.stop_stream()
	stream.close()
	p.terminate()
	close_files()
	print('catched interrupt signal!')
	sys.exit(0)


# 注册ctrl-c中断
signal.signal(signal.SIGINT, sigint_handler)

print("Number of devices: ", p.get_device_count())

device_index = -1

for index in range(0, p.get_device_count()):
	info = p.get_device_info_by_index(index)
	device_name = info.get("name")
	print("device_name: ", device_name)
	if device_name.find(RECORD_DEVICE_NAME) != -1:
		device_index = index
		break

if device_index != -1:
	print("find the device")
	stream.close()

	print(p.get_device_info_by_index(device_index))

	stream = p.open(
		rate=RECORD_RATE,
		format=p.get_format_from_width(RECORD_WIDTH),
		channels=RECORD_CHANNELS,
		input=True,
		input_device_index=device_index,
		start=False)
else:
	print("don't find the device")

record()