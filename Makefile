CXX = g++
CXXFLAGS = -Wall -std=c++23 -g -Iinclude -Ithirdparty

BUILD_DIR = build

TARGET = $(BUILD_DIR)/main.out

OBJS = $(BUILD_DIR)/main.o $(BUILD_DIR)/image_filter.o

$(shell mkdir -p $(BUILD_DIR))

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

# 编译 main.o
$(BUILD_DIR)/main.o: test/main.cpp include/image_filter.h
	$(CXX) $(CXXFLAGS) -c test/main.cpp -o $(BUILD_DIR)/main.o

# 编译 image_filter.o
$(BUILD_DIR)/image_filter.o: src/image_filter.cpp include/image_filter.h
	$(CXX) $(CXXFLAGS) -c src/image_filter.cpp -o $(BUILD_DIR)/image_filter.o

# 清理
clean:
	rm -rf $(BUILD_DIR)

# 运行
run: $(TARGET)
	./$(TARGET)

.PHONY: clean run