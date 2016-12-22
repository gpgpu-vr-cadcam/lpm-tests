NVCC=nvcc
SOURCE=../../Source
BIN=../../Bin
NVCC_FLAGS=-rdc=true -cudart static -lcudart -gencode arch=compute_52,code=sm_52

GTEST=/usr/lib/libgtest.a  /usr/lib/libgtest_main.a

UTILS=$(BIN)/Tmp/Utils.o
CORE=$(BIN)/Tmp/IPSet.o
MATCHERS=$(BIN)/Tmp/TreeMatcher.o
LIB=$(CORE) $(UTILS) $(MATCHERS)


runner: lib $(BIN)/Runner

$(BIN)/Runner: $(LIB) $(BIN)/Tmp/Runner.o
	$(NVCC) $(NVCC_FLAGS) -lpthread -o $(BIN)/Runner $(LIB) $(BIN)/Tmp/Runner.o

$(BIN)/Tmp/Runner.o: $(SOURCE)/Runners/Runner.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(SOURCE)/Runners/Runner.cpp -o $(BIN)/Tmp/Runner.o

lib: bin $(LIB)

$(BIN)/Tmp/TreeMatcher.o: $(SOURCE)/Matchers/TreeMatcher.cu $(SOURCE)/Matchers/TreeMatcher.cuh $(SOURCE)/Core/BaseResult.h
	$(NVCC) $(NVCC_FLAGS) -c $(SOURCE)/Matchers/TreeMatcher.cu -o $(BIN)/Tmp/TreeMatcher.o

$(BIN)/Tmp/IPSet.o: $(SOURCE)/Core/IPSet.cu $(SOURCE)/Core/IPSet.cuh
	$(NVCC) $(NVCC_FLAGS) -c $(SOURCE)/Core/IPSet.cu -o $(BIN)/Tmp/IPSet.o

$(BIN)/Tmp/Utils.o: $(SOURCE)/Utils/Utils.cpp $(SOURCE)/Utils/Utils.h
	$(NVCC) $(NVCC_FLAGS) -c $(SOURCE)/Utils/Utils.cpp -o $(BIN)/Tmp/Utils.o

bin:
	mkdir -p ../../Bin
	mkdir -p ../../Bin/Tmp

clear: clean
	rm -f $(BIN)/*.o
	rm -r -f $(BIN)

clean:
	rm -f $(BIN)/Tmp/*.o
	rm -r -f $(BIN)/Tmp

.PHONY: all clear clean
