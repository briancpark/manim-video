"""Narration scripts. Two products:

  SHORT  : one voiceover block over the vertical Short.mp4 (~30s).
  LONG   : per-scene narration for the horizontal explainer.

Each LONG entry maps a rendered Manim scene to its narration. The builder
synthesizes the audio, then fits the clip length to max(video, narration).
Keep narration close to each scene's animation length to minimize freeze-pad.
"""

# (scene_module, ClassName)  ->  used to locate media/videos/<module>/<q>/<Class>.mp4
SHORT = {
    "scene": ("conv_short", "Short"),
    "text": (
        "This is a convolution, the core of every image filter and neural network. "
        "The naive version is six nested loops, painfully slow. "
        "The trick: vectorize across the width. "
        "Take one weight, broadcast it into four lanes, "
        "and multiply it against four pixels at once, in a single fused multiply-add. "
        "Keep sixteen in flight and the pipeline never stalls. "
        "Same math, bit for bit identical, but fourteen times faster on Apple Silicon. "
        "That's NEON SIMD."
    ),
}

LONG = [
    {
        "scene": ("conv_anim", "NaiveConv"),
        "text": (
            "Let's make a convolution fast. A convolution slides a small kernel over "
            "an image. For every output pixel, we multiply the kernel against the patch "
            "underneath it and sum the results. The textbook way is six nested loops, "
            "computing one output pixel completely before moving to the next. "
            "It's simple and correct, but slow, because every weight is read from memory "
            "again and again, and nothing stays in registers. This is our baseline."
        ),
    },
    {
        "scene": ("conv_anim", "NeonConv"),
        "text": (
            "Here's the key idea. Neighboring output pixels along a row share the same "
            "weights and read adjacent input pixels. So instead of finishing one pixel "
            "at a time, we vectorize across the width. We take a single weight, broadcast "
            "it into all four lanes of a NEON register, and apply it to four output pixels "
            "at once with a fused multiply-add. To hide the latency of each instruction, "
            "we keep sixteen output pixels in flight at the same time, so the processor's "
            "pipeline is always full. Same arithmetic, but now it's bound by compute "
            "throughput, not by waiting."
        ),
    },
    {
        "scene": ("conv_im2col", "Im2Col"),
        "text": (
            "There's another path that the fastest libraries take, called im to col. "
            "The idea is to flatten the problem into a matrix multiply. Each sliding "
            "patch of the image is unrolled into a single column of a big matrix. "
            "Every column holds one output pixel's receptive field, laid out contiguously. "
            "We trade memory for regularity, turning an awkward sliding window into a "
            "dense, predictable layout."
        ),
    },
    {
        "scene": ("conv_im2col", "Im2ColGEMM"),
        "text": (
            "Once the image is lowered into that matrix, the filters become rows of a "
            "weight matrix, and the entire convolution is just one matrix multiply: "
            "weights times the im to col matrix. Each output value is a dot product of a "
            "filter row and an image column. This is why convolution is so fast in "
            "practice: it inherits decades of hand-tuned matrix multiply kernels in BLAS, "
            "cuDNN, and oneDNN. Finally, we reshape each row of the result back into a "
            "two-dimensional feature map."
        ),
    },
    {
        "scene": ("conv_channels", "ChannelReduce"),
        "text": (
            "Real images have multiple channels, like red, green, and blue. Now each "
            "output pixel sums a partial convolution from every input channel. This sum "
            "over input channels is a reduction: a dependency chain that has to "
            "accumulate before the pixel is done. It's the part you cannot simply split "
            "apart without combining the results."
        ),
    },
    {
        "scene": ("conv_channels", "ChannelParallel"),
        "text": (
            "But the output channels tell a different story. Each filter produces its own "
            "independent output plane, all reading from the same input. That makes the "
            "output channels embarrassingly parallel: you can map them across threads, "
            "cores, and SIMD lanes with no synchronization at all. So the recipe for a "
            "fast convolution is this: reduce over input channels, and run the output "
            "channels in parallel. Same math, many times faster. That's high-performance "
            "computing."
        ),
    },
]
