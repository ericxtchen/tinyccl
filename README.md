# tinyccl

So far, this code simulates multiple GPUs as multiple processes that use my single RTX 4060 GPU on my laptop where chunks are sent between them using TCP.

 Benchmarked results for the code on an RTX 4060 are in [results.png](results.png).


Hopefully, I can find a way to get multiple GPUs to measure P2P memory access with `cudaMemcpyPeerAsync`...