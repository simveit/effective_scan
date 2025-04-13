NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_LDFLAGS = -lcublas -lcuda
OUT_DIR = out

CUDA_OUTPUT_FILE = -o $(OUT_DIR)/$@
NCU_PATH := $(shell which ncu)
NCU_COMMAND = sudo $(NCU_PATH) --set full --import-source yes

NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCC_FLAGS += -arch=sm_90a

NVCC_BASE = nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo

kernel_0: kernel_0.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

kernel_1: kernel_1.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

kernel_2: kernel_2.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

kernel_3: kernel_3.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

kernel_4: kernel_4.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

compile_all: 
	make kernel_0
	make kernel_1
	make kernel_2
	make kernel_3
	make kernel_4

run_all: 
	./$(OUT_DIR)/kernel_0
	./$(OUT_DIR)/kernel_1
	./$(OUT_DIR)/kernel_2
	./$(OUT_DIR)/kernel_3
	./$(OUT_DIR)/kernel_4

clean:
	rm $(OUT_DIR)/*