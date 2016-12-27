NVCC=nvcc
SOURCE=../../Source
BIN=../../Bin
NVCC_FLAGS=-rdc=true -cudart static -lcudart -gencode arch=compute_20,code=sm_20 -std=c++11

GTEST=/usr/lib/libgtest.a  /usr/lib/libgtest_main.a

UTILS=$(BIN)/Tmp/Utils.o
CORE=$(BIN)/Tmp/IPSet.o
MATCHERS=$(BIN)/Tmp/TreeMatcher.o
LIB=$(CORE) $(UTILS) $(MATCHERS)

GTEST=/usr/lib/libgtest.a  /usr/lib/libgtest_main.a
PERF_TEST=$(BIN)/Tmp/PerformanceTests.o $(BIN)/Tmp/PerformanceTestsMain.o
BASIC_TEST=$(BIN)/Tmp/IPSetTests.o $(BIN)/Tmp/TreeMatcherTests.o $(BIN)/Tmp/TestsMain.o
TEST_RUNNERS=$(BIN)/PerformanceTests $(BIN)/BasicTests

all: lib test

test: bin $(LIB) $(PERF_TEST) $(BASIC_TEST) $(TEST_RUNNERS)

lib: bin $(LIB)

$(BIN)/BasicTests: $(LIB) $(BASIC_TEST)
	$(NVCC) $(NVCC_FLAGS) -lpthread -o $(BIN)/BasicTests $(LIB) $(BASIC_TEST) $(GTEST)

$(BIN)/Tmp/TestsMain.o: $(SOURCE)/Tests/TestsMain.cu $(SOURCE)/Tests/Tests.h
	$(NVCC) $(NVCC_FLAGS) -c $(SOURCE)/Tests/TestsMain.cu -o $(BIN)/Tmp/TestsMain.o

$(BIN)/Tmp/TreeMatcherTests.o: $(SOURCE)/Tests/TreeMatcherTests.cpp $(SOURCE)/Tests/Tests.h
	$(NVCC) $(NVCC_FLAGS) -c $(SOURCE)/Tests/TreeMatcherTests.cpp -o $(BIN)/Tmp/TreeMatcherTests.o

$(BIN)/Tmp/IPSetTests.o: $(SOURCE)/Tests/IPSetTests.cpp $(SOURCE)/Tests/Tests.h
	$(NVCC) $(NVCC_FLAGS) -c $(SOURCE)/Tests/IPSetTests.cpp -o $(BIN)/Tmp/IPSetTests.o

$(BIN)/PerformanceTests: $(LIB) $(PERF_TEST)
	$(NVCC) $(NVCC_FLAGS) -lpthread -D PERF_TEST -o $(BIN)/PerformanceTests $(LIB) $(PERF_TEST) $(GTEST)

$(BIN)/Tmp/PerformanceTestsMain.o: $(SOURCE)/Tests/PerformanceTestsMain.cu $(SOURCE)/Tests/Tests.h
	$(NVCC) $(NVCC_FLAGS) -D PERF_TEST -c $(SOURCE)/Tests/PerformanceTestsMain.cu -o $(BIN)/Tmp/PerformanceTestsMain.o

$(BIN)/Tmp/PerformanceTests.o: $(SOURCE)/Tests/PerformanceTests.cpp $(SOURCE)/Tests/Tests.h
	$(NVCC) $(NVCC_FLAGS) -D PERF_TEST -c $(SOURCE)/Tests/PerformanceTests.cpp -o $(BIN)/Tmp/PerformanceTests.o

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
