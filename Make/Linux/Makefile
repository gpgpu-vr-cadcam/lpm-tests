NVCC=nvcc
SOURCE=../../Source
BIN=../../Bin
NVCC_FLAGS=-rdc=true -cudart static -gencode arch=compute_52,code=sm_52

GTEST=/usr/lib/libgtest.a  /usr/lib/libgtest_main.a



Bin:
	mkdir -p ../../Bin
	mkdir -p ../../Bin/Tmp

clear: clean
	rm -f $(BIN)/*.o
	rm -r -f $(BIN)

clean:
	rm -f $(BIN)/Tmp/*.o
	rm -r -f $(BIN)/Tmp

.PHONY: all clear clean
