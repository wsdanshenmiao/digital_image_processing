CXX = g++
CXXFLAGS = -Wall -std=c++23 -g -Iinclude -Ithirdparty

BUILD_DIR = build

TARGET = $(BUILD_DIR)/main.out

OBJS = $(BUILD_DIR)/main.o

$(shell mkdir -p $(BUILD_DIR))

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

# 编译 main.o
$(BUILD_DIR)/main.o: test/main.cpp include/image_filter.h
	$(CXX) $(CXXFLAGS) -c test/main.cpp -o $(BUILD_DIR)/main.o

# 清理
clean:
	rm -rf $(BUILD_DIR)

# 运行
run: $(TARGET)
	./$(TARGET) $(args)

.PHONY: clean run