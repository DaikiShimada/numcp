CC = nvcc
CFLAGS = -O3 -std=c++11

# src directory
DIRS := example

.PHONY: all install $(DIRS)

all: libnumcpcuda.so $(DIRS)

libnumcpcuda.so : src/numcp_cuda.cpp
	$(CC) $(CFLAGS) -Iinclude -lcublas -shared -Xcompiler -fPIC -o libnumcpcuda.so src/numcp_cuda.cpp

install : libnumcpcuda.so
	cp -r include/numcp /usr/local/include/
	cp libnumcpcuda.so /usr/local/lib/ 

$(DIRS):
	$(MAKE) -C $@
