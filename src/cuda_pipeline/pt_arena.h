/*
 * pt_arena.h - GPU arena allocation API.
 *
 * One cudaMalloc for all GPU buffers, one cudaMallocHost for all pinned host
 * buffers.  Every pointer in PT_GpuArena is assigned as an offset into the
 * single base allocation.  No per-buffer allocations, no fragmentation.
 */

#ifndef PT_ARENA_H
#define PT_ARENA_H

#include "pt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * pt_arena_init - Allocate all GPU and pinned host memory for the pipeline.
 *
 * Computes buffer sizes from the session parameters (camera count, frame
 * dimensions, batch size), performs ONE cudaMalloc and ONE cudaMallocHost,
 * and assigns every pointer in `arena` as an offset from the base.
 *
 * Returns PT_OK on success, PT_ERR_CUDA or PT_ERR_OUT_OF_MEMORY on failure.
 */
int pt_arena_init(PT_GpuArena *arena,
                  int num_cameras,
                  int frame_width,
                  int frame_height,
                  int batch_size);

/*
 * pt_arena_destroy - Free the GPU and pinned host allocations.
 *
 * Safe to call on a zeroed arena (no-op).
 */
void pt_arena_destroy(PT_GpuArena *arena);

/*
 * pt_arena_print_stats - Print total GPU and host allocation sizes to stdout.
 */
void pt_arena_print_stats(const PT_GpuArena *arena);

#ifdef __cplusplus
}
#endif

#endif /* PT_ARENA_H */
