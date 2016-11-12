all: psum-serial.c psum-naive.cu
	gcc psum-serial.c -g -o psum-serial
	nvcc psum-naive.cu -g -o psum-naive

serial: psum-serial.c
	gcc psum-serial.c -g -o psum-serial

naive: psum-naive.cu
	nvcc psum-naive.cu -g -o psum-naive

clean:
	rm psum-serial
	rm psum-naive
