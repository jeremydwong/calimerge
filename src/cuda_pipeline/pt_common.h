/*
 * pt_common.h - CUDA pipeline wrapper.
 *
 * Defines PT_CUDA_PIPELINE to enable CUDA-specific structs (PT_GpuArena,
 * PT_SyncTable, PT_PipelineConfig, PT_Stats), then includes the shared header.
 *
 * All CUDA pipeline files include this header (unchanged #include "pt_common.h").
 */

#ifndef PT_COMMON_CUDA_H
#define PT_COMMON_CUDA_H

#define PT_CUDA_PIPELINE
#include "../pt_shared/pt_common.h"

#endif /* PT_COMMON_CUDA_H */
