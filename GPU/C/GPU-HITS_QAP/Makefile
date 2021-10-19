################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= 	PITSQAP

# CUDA source files (compiled with cudacc)
CUFILES		:= \
			Meta.cu\
			GPU_TS.cu\
			CPU_TS.cu\
			GPU_Evaluation_QAP.cu\
			CPU_Evaluation_QAP.cu

# CUDA dependency files
CU_DEPS		:= \
			Meta.cuh\
			GPU_TS.cuh\
			CPU_TS.cuh\
			GPU_Evaluation_QAP.cuh\
			CPU_Evaluation_QAP.cuh

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= \
			main.cpp\
			data.cpp

# C/C++ dependency files
C_DEPS :=     \
			data.h

# Additional compiler flags and LIBs to include
USEGLLIB          := 0
USEGLUT           := 0
USECURAND         := 1

ptxas := 0

SMVERSIONFLAGS	:= -arch sm_13
NVCCFLAGS += --compiler-options "-O3" 

################################################################################
# Rules and targets

MPICXX = $(shell which mpicxx)
ifeq ($(strip $(MPICXX)),)
	NO_MPI := 1
endif

ifdef NO_MPI
all:
	@echo "MPI not found, not building simpleMPI."
clean:
tidy:
clobber:
else
LIBFLAGS += -I/usr/include/openmpi
include ~/NVIDIA_GPU_Computing_SDK/C/common/common.mk

endif


$(OBJDIR)/%.cpp.o : $(SRCDIR)%.cpp $(C_DEPS)
	$(VERBOSE)mpicxx.openmpi $(CXXFLAGS) $(LIBFLAGS) -o $@ -c $<

# This option for Mac allows CUDA applications to work without requiring to set DYLD_LIBRARY_PATH
ifneq ($(DARWIN),)
   LDFLAGS += -Xlinker -rpath $(CUDA_INSTALL_PATH)/lib
endif

#LINKLINE  = nvcc $(LDFLAGS) $(LIBFLAGS) -o $(TARGET) $(OBJS) $(LIB)
#LINKLINE  = g++ -fPIC $(LDFLAGS) $(LIBFLAGS) -o $(TARGET) $(OBJS) $(LIB)
LINKLINE  = mpicxx.openmpi $(CXX_ARCH_FLAGS) -fPIC $(LDFLAGS) $(LIBFLAGS) -o $(TARGET) $(OBJS) $(LIB)

#LINKLINE  = mpicxx $(CXX_ARCH_FLAGS) -fPIC $(LDFLAGS) $(LIBFLAGS) -o $(TARGET) $(OBJS) $(LIB)

#test
#nvcc main.cpp data.cpp Meta.cu GPU_TS.cu CPU_TS.cu GPU_Evaluation_QAP.cu CPU_Evaluation_QAP.cu -o PITSQAP
