CC=g++
LD=ld
CXXFLAGS=-Wall -std=c++11 -g -O
INCS=-I"/usr/local/cuda/include" -I"/usr/local/TensorRT-7.1.3.4/include" -I"/usr/local/include" -I"plugin"
LIBS=-L"/usr/local/cuda/lib64" -L"/usr/local/TensorRT-7.1.3.4/lib" -L"/usr/local/lib" -Wl,--start-group -lnvinfer -lnvparsers -lnvinfer_plugin -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart -lrt -ldl -lpthread -Wl,--end-group

NVCC=nvcc
NVCCFLAGS=-m64 -gencode arch=compute_53,code=sm_53 \
               -gencode arch=compute_62,code=sm_62 \
               -gencode arch=compute_62,code=compute_62

.PHONY: all clean

all: libyolo_layer.so

clean:
	rm -f *.so *.o

libyolo_layer.so: yolo_layer.o
	$(CC) -shared -o $@ $< $(LIBS)

yolo_layer.o: yolo_layer.cu yolo_layer.h
	$(NVCC) -ccbin $(CC) $(INCS) $(NVCCFLAGS) -Xcompiler -fPIC -c -o $@ $<
