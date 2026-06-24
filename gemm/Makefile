CC      ?= cc
CFLAGS  ?= -O3 -ffast-math -march=native -std=c11 -Wall -Wextra
# Apple Accelerate gives a tuned BLAS to benchmark against.
CFLAGS  += -DUSE_ACCELERATE
LDLIBS  := -lm -framework Accelerate

OBJ := gemm_naive.o gemm_blocked.o gemm_neon.o main.o

gemm: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDLIBS)

%.o: %.c gemm.h
	$(CC) $(CFLAGS) -c -o $@ $<

run: gemm
	./gemm

# Manim animations (dedicated Manim Community env).
MANIM   := /Users/briancpark/miniforge3/envs/manim-ce/bin/manim
MANIM_Q ?= -qh
anim:
	$(MANIM) $(MANIM_Q) --format mp4 gemm_anim.py Tiling
	$(MANIM) $(MANIM_Q) --format mp4 gemm_anim.py Microkernel
	$(MANIM) $(MANIM_Q) --format mp4 gemm_anim.py Roofline

clean:
	rm -f gemm $(OBJ)

.PHONY: run anim clean
