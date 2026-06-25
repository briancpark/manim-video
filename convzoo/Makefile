# The Convolution Zoo — animations + narrated video.
MANIM   := /Users/briancpark/miniforge3/envs/manim-ce/bin/manim
MANIM_Q ?= -qh
SCENES  := ColdOpen SpatialFamily GemmUnify ChannelZoo ShapeTricks \
           NEON SME TensorCore WhyMatrixWins GenAI
VOICE   ?= kokoro

anim:
	$(MANIM) $(MANIM_Q) --format mp4 convzoo_anim.py $(SCENES)

long:
	/opt/homebrew/bin/python3 ../pipeline/build.py convzoo long $(VOICE)

clean:
	rm -rf media

.PHONY: anim long clean
