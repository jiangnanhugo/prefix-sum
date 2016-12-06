all: psum-serial.c psum-naive.cu
	gcc psum-serial.c -g -o psum-serial
	nvcc psum-naive.cu -g -o psum-naive
	nvcc psum-shmem.cu -g -o psum-shmem

serial: psum-serial.c
	gcc psum-serial.c -g -o psum-serial

naive: psum-naive.cu
	nvcc psum-naive.cu -g -o psum-naive

shmem: psum-shmem.cu
	nvcc psum-shmem.cu -g -o psum-shmem

clean:
	rm psum-serial
	rm psum-naive
	rm psum-shmem
