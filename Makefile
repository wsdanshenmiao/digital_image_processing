USE_GPU ?= 0

ifeq ($(USE_GPU), 0)

CXX = g++
CXXFLAGS = -Wall -std=c++23 -g -Iinclude -Ithirdparty

SRC = test/cpu_test.cpp $(wildcard src/*.cpp)
BUILD_DIR = build

TARGET = $(BUILD_DIR)/cpu_test.out

OBJS = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SRC))

else

CXX = nvcc
CXXFLAGS = -g -Iinclude -Ithirdparty --extended-lambda

SRC = test/gpu_test.cu $(wildcard src/cuda/*.cu)
BUILD_DIR = build_gpu

TARGET = $(BUILD_DIR)/gpu_test.out

OBJS = $(patsubst %.cu, $(BUILD_DIR)/%.o, $(SRC))

endif

$(shell mkdir -p $(BUILD_DIR))

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

# 编译
ifeq ($(USE_GPU), 0)
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@
else
$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@
endif

# 清理
clean:
	rm -rf $(BUILD_DIR) build build_gpu

# 运行
run: $(TARGET)
	./$(TARGET) $(ARGS)

.PHONY: clean run