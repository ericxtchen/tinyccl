TinyCCL is an experimental collective communication library implemented in C++ and CUDA. It explores ring all-reduce using interchangeable CPU and GPU computation backends and TCP-based interprocess communication.

The current experimental setup executes multiple ranks as separate processes sharing a single RTX 4060 GPU. Benchmarks characterize collective execution time, memory staging, communication, and computation overhead.

Multi-GPU peer-to-peer communication and interconnect performance remain future extensions due to hardware availability.
