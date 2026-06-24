CC      ?= cc
CFLAGS  ?= -O3 -ffast-math -march=native -std=c11 -Wall -Wextra
LDLIBS  := -lm

COMMON := conv2d_naive.o conv2d_neon.o

all: conv2d conv_filter

conv2d: $(COMMON) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

conv_filter: $(COMMON) filter.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c conv2d.h
	$(CC) $(CFLAGS) -c -o $@ $<

run: conv2d
	./conv2d

# Benchmark sweep + plots (perf chart and Sobel/blur feature maps).
viz: conv2d conv_filter
	./conv2d --sweep > sweep.csv
	python3 visualize.py

# Manim animations of the algorithm (dedicated Manim Community env).
MANIM := /Users/briancpark/miniforge3/envs/manim-ce/bin/manim
MANIM_Q ?= -ql        # -ql low / -qh high / -qk 4k

anim:
	$(MANIM) $(MANIM_Q) --format mp4 conv_anim.py NaiveConv
	$(MANIM) $(MANIM_Q) --format mp4 conv_anim.py NeonConv

# im2col + GEMM lowering (conv as a single matrix multiply).
anim-im2col:
	$(MANIM) $(MANIM_Q) --format mp4 conv_im2col.py Im2Col
	$(MANIM) $(MANIM_Q) --format mp4 conv_im2col.py Im2ColGEMM

# Multi-channel: Cin reduction vs Cout parallelism (RGB example).
anim-channels:
	$(MANIM) $(MANIM_Q) --format mp4 conv_channels.py ChannelReduce
	$(MANIM) $(MANIM_Q) --format mp4 conv_channels.py ChannelParallel

# Narrated videos with on-device TTS + ducked music. VOICE=kokoro|say
VOICE ?= kokoro
short:
	/opt/homebrew/bin/python3 video/build.py short $(VOICE)
long:
	/opt/homebrew/bin/python3 video/build.py long $(VOICE)

clean:
	rm -f conv2d conv_filter $(COMMON) main.o filter.o sweep.csv *.png /tmp/conv_*.bin

.PHONY: all run viz anim anim-im2col anim-channels short long clean
