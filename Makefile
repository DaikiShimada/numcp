CXX = g++
CXXFLAGS = -O0 -std=c++11
LIBS = -lglog -lcblas
INCLUDE = -Iinclude
HEADER_DIR := include
SRC_HEAD := $(wildcard $(HEADER_DIR)/*.hpp)
EXAMPLE_DIR := example
EXAMPLE_SRC := $(wildcard $(EXAMPLE_DIR)/*.cpp)
EXAMPLE_BIN := $(EXAMPLE_SRC:.cpp=.bin)

.PHONY: all 

all : $(EXAMPLE_BIN)

# make example
$(EXAMPLE_DIR)/%.bin : $(EXAMPLE_DIR)/%.cpp $(SRC_HEAD)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $< $(LIBS)
