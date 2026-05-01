# TODO

## Offline pipeline unification
- [ ] Add multi-sync-index `process_batch(frame_lists[B], sync_indices[B])`
      entry to `pt_stream` (C++/CUDA TRT) and `pt_stream_mps` (Obj-C++/CoreML)
      so the unified offline worker can submit batches > 1 per call. Without
      it, offline CUDA / MPS lose throughput vs the deprecated `pt_main.cpp`
      batched path. Phase 2 of the unification.
- [ ] Empirically determine the optimal `batch_size` for the unified offline
      pipeline once the C-side batch entry exists. Test
      {1, 2, 4, 8, 16, 32}. Power-of-two probably matters for warp efficiency
      on Ampere / Ada but is not guaranteed; measure.
- [ ] Once the unified worker has been validated on real recordings on both
      CUDA and MPS, delete `pt_main.cpp`, `run_cuda_pipeline`,
      `run_mps_pipeline`, and `OfflineProcessingWorker`.

## Offline reprocessing visualisation + regression
- [ ] Re-run the offline reprocessing test with these additions to the
      ankle-positions plot:
      1. Put the offline reprocessing duration in the title of the
         output figure.
      2. Save a regression `.pkl` snapshot of the offline run alongside
         the npz so future code changes can be diffed against a known-
         good baseline.
      3. Investigate what actually happens around t = 12 s — is that
         where the user turns around? Worth understanding why ankle
         position takes a discontinuity there.
      4. Add a fourth subplot showing the framerate (instantaneous +
         rolling mean) alongside ankle x/y/z so dropped frames
         correlate visually with anomalous-position events.
