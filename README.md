# A simple counting script for estimating llama3 inference time

Run with
```
python projection.py
```

The output will show 
1. KV-cache size. The total memory consumed by the KV-cache.
2. Time. The estimated total inference time.
3. T/s. The estimated tokens per second.
4. Prefill time. The total time spent processing the prompt batch.
5. Next time. The total time spent on single generation of a batch.
6. Generation time. The total time spent generating all tokens.
7. Gen:Prefill ratio. The ratio of time spent on generation compared to prefill.
8. Compute/Memory ratio. The ratio of compute operations to memory operations during generation (used for correction).
