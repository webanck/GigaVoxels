#ifndef _MACRO_H_
#define _MACRO_H_

// Variables used for the tests.
#define N_BRICKS 2000
#define MAX_COMPRESSION 250

// Size of the blocs of data used to compress (no link with the GigaVoxels bricks).
#define BRICK_SIZE 250
#if BRICK_SIZE >= UCHAR_MAX
#error "BRICK_SIZE is too big"
#endif

// Max size for the compressed brick (if it's bigger, we gain nothing by compressing it). 
#define MAX_COMPRESSED_BRICK_SIZE (( BRICK_SIZE * sizeof( unsigned int )) / ( sizeof( unsigned int ) + sizeof( unsigned char ) ))


// Size of the array containing compressed data.
#define STR_SIZE N_BRICKS * BRICK_SIZE

// Variables managing the number of threads.
#define GRID_SIZE 1000
#define N_THREADS_PER_BLOCS 192
#define WARP_SIZE 32
#define N_WARPS_PER_BLOCS N_THREADS_PER_BLOCS / WARP_SIZE

#endif // _MACRO_H_
