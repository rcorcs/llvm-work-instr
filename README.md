# Work Profiling
Low-overhead instrumentation for work-based profiling in LLVM

This implementation uses LLVM's built-in cost model in order to compute the work metric.

The work metric considers that the work of a basic block is the sum of the cost of its LLVM instructions
and the final work of a program's execution is the linear combination of basic block frequency and their work value.
