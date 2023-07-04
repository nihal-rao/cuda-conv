# cuda-conv

An implementation of 2d convolutions using shared memory tiling. Achieves 1.5x speedup over naive cuda kernel and 560x speedup over cpu implementation.

## Implementation details
* Uses shared memory block of 32*32.
* Constant memory is used to aggressively cache the filter.
* Input 2D matrix is randomly initialised with values in [0,255] and the filter is initialised with floats in the range [-1, 1].
* No halo (padding) cells used - if input is size (H, W), output will be size (H-K+1, W-K+1) where K is filter size.
* Error checking is done for each of the cuda kernels versus the cpu version.
* All calculations are done in single precision floating point.

## Usage
* First run ```make``` command to build object files and the main executable, ```program```.
* Object files are built in the `build` directory.
* Run the executable as ```./program H W K``` where H, W are input dimensions and K is square filter length (must be odd and less than 16).

