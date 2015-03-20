
OBJS += \
bilateral_kernel.o

OBJS2 += \
L3.o

%.o: %.cu
	$(NVCC) $(CFLAGS) $(EXTRA_CFLAGS) -c -o "$@" "$<"
