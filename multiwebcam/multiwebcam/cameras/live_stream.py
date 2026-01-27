# This widget is the primary functional unit of the motion capture. It
# establishes the connection with the video source and manages the thread
# that reads in frames.


import multiwebcam.logger

from time import perf_counter, sleep
from queue import Queue
from threading import Thread, Event

import cv2, numpy as np
from multiwebcam.cameras.camera import Camera
from multiwebcam.interface import FramePacket

logger = multiwebcam.logger.get(__name__)

class LiveStream():
    def __init__(self, camera: Camera, fps_target: int = 6):
        self.camera: Camera = camera
        self.port = camera.port

        self.stop_event = Event()

        # list of queues that will have frame packets pushed to them
        self.subscribers = []

        # make sure camera no longer reading before trying to change resolution
        self.stop_confirm = Queue()

        self._show_fps = False  # used for testing


        self.set_fps_target(fps_target)
        self.FPS_actual = 0
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self._play_worker, args=(), daemon=True)
        self.thread.start()

        # initialize time trackers for actual FPS determination
        self.frame_time = perf_counter()
        self.avg_delta_time = 1  # initialize to something to avoid errors elsewhere

    @property
    def size(self):
        # because the camera resolution will potentially change after stream initialization, this should be 
        # read directly from the camera whenever a caller (e.g. videorecorder) wants the current resolution
        return self.camera.size

    def subscribe(self, queue: Queue):
        if queue not in self.subscribers:
            logger.info(f"Adding queue to subscribers at stream {self.port}")
            self.subscribers.append(queue)
            logger.info(f"...now {len(self.subscribers)} subscriber(s) at {self.port}")
        else:
            logger.warn(
                f"Attempted to subscribe to live stream at port {self.port} twice"
            )

    def unsubscribe(self, queue: Queue):
        try:
            if queue in self.subscribers:
                logger.info(f"Removing subscriber from queue at port {self.port}")
                self.subscribers.remove(queue)
                logger.info(
                    f"{len(self.subscribers)} subscriber(s) remain at port {self.port}"
                )
            else:
                logger.warn(
                    f"Attempted to unsubscribe to live stream that was not subscribed to\
                    at port {self.port} twice"
                )
        except:
            logger.warn("Attempted to remove queue that may have been removed twice at once")

    def set_fps_target(self, fps_target):
        """
        This is done through a method as it will also do a one-time determination of the times as which
        frames should be read (the milestones)
        """

        self.fps_target = fps_target
        self.target_interval = 1 / fps_target
        milestones = []
        for i in range(0, fps_target):
            milestones.append(i / fps_target)
        logger.info(f"Setting fps to {self.fps_target} at port {self.port}")
        self.milestones = np.array(milestones)

    def wait_to_next_frame(self):
        """
        based on the next milestone time, return the time needed to sleep so that
        a frame read immediately after would occur when needed
        """

        time = perf_counter()
        fractional_time = time % 1
        all_wait_times = self.milestones - fractional_time
        future_wait_times = all_wait_times[all_wait_times > 0]

        if len(future_wait_times) == 0:
            return 1 - fractional_time
        else:
            return future_wait_times[0]

    def get_FPS_actual(self):
        """
        set the actual frame rate; called within roll_camera()
        needs to be called from within roll_camera to actually work
        Note that this is a smoothed running average
        """
        self.delta_time = perf_counter() - self.start_time
        self.start_time = perf_counter()
        if not self.avg_delta_time:
            self.avg_delta_time = self.delta_time

        # folding in current frame rate to trailing average to smooth out
        self.avg_delta_time = 0.5 * self.avg_delta_time + 0.5 * self.delta_time
        self.previous_time = self.start_time
        return 1 / self.avg_delta_time

    def _play_worker2(self):
        logger.info(f"Port {self.port}: _play_worker thread started.")
        self.frame_index = 0
        self.start_time = perf_counter()  # For your existing FPS_actual calculation

        # Initialize next_frame_due_time
        # Start so the first frame processing begins almost immediately or after one interval
        next_frame_due_time = perf_counter() + self.target_interval 
        logger.info(f"Port {self.port}: Initial next_frame_due_time: {next_frame_due_time:.4f}")

        # For periodic detailed logging, not every frame
        log_interval_frames = self.fps_target * 5 # e.g., Log detailed timing every 5 seconds
        if log_interval_frames == 0: log_interval_frames = 30 # Default if fps_target is 0 or low

        first_loop_iteration = True
        while not self.stop_event.is_set():
            if first_loop_iteration:
                logger.info(f"Port {self.port}: Camera now rolling, entering main loop.")
                first_loop_iteration = False

            current_time_before_sleep = perf_counter()
            sleep_duration = next_frame_due_time - current_time_before_sleep

            if sleep_duration > 0.0005: # Sleep if duration is more than 0.5ms
                if self.frame_index % log_interval_frames == 0:
                    logger.debug(f"Port {self.port}: Frame {self.frame_index}. Sleeping for {sleep_duration:.4f}s. Due at {next_frame_due_time:.4f}, Current: {current_time_before_sleep:.4f}")
                sleep(sleep_duration)
            elif sleep_duration < -self.target_interval: # If we're already late by more than one frame interval
                logger.warning(f"Port {self.port}: Frame {self.frame_index}. Significantly late by {-sleep_duration:.4f}s. Due at {next_frame_due_time:.4f}, Current: {current_time_before_sleep:.4f}")


            # ----- CRITICAL: Update for the *next* iteration's due time -----
            # Schedule the *next* frame relative to when the *current* one was due.
            scheduled_current_frame_due_time = next_frame_due_time # Store for logging this iteration
            next_frame_due_time += self.target_interval

            # ----- Safety: Prevent runaway accumulation of lateness -----
            current_time_after_potential_sleep = perf_counter()
            # If the *new* next_frame_due_time is already in the past by too much
            if next_frame_due_time < current_time_after_potential_sleep - (self.target_interval * 2): # Tunable factor (e.g., 2 frame intervals)
                old_next_due = next_frame_due_time
                next_frame_due_time = current_time_after_potential_sleep + self.target_interval
                logger.warning(f"Port {self.port}: Frame {self.frame_index}. Significant lag. Resetting frame schedule from {old_next_due:.4f} to {next_frame_due_time:.4f}")

            if self.camera.capture.isOpened():
                # Spinlock for subscribers
                spinlock_looped = False
                # This spinlock should ideally be short. If it's long, it impacts timing.
                # Consider if it should be before or after sleep calculation depending on typical wait time.
                while len(self.subscribers) == 0 and not self.stop_event.is_set():
                    if not spinlock_looped:
                        logger.info(f"Port {self.port}: Frame {self.frame_index}. Spinlock initiated (no subscribers).")
                        spinlock_looped = True
                    sleep(0.1) # Reduced sleep in spinlock from 0.5 for quicker recovery
                if spinlock_looped:
                    logger.info(f"Port {self.port}: Frame {self.frame_index}. Spinlock released (subscribers present or stopping).")

                if self.stop_event.is_set(): # Check again after potential spinlock sleep
                    break

                grab_t_start = perf_counter()
                grab_success = self.camera.capture.grab()
                grab_t_end = perf_counter()
                grab_duration_ms = (grab_t_end - grab_t_start) * 1000

                if not grab_success:
                    logger.warning(f"Port {self.port}: Frame {self.frame_index}. camera.grab() FAILED. Grab duration: {grab_duration_ms:.2f}ms.")
                    self.success = False
                    self.frame = None
                    # Optionally: consider a short delay to let camera recover before next attempt
                    # sleep(0.01) # This would add to the cycle time
                else:
                    retrieve_t_start = perf_counter()
                    self.success, self.frame = self.camera.capture.retrieve()
                    retrieve_t_end = perf_counter()
                    retrieve_duration_ms = (retrieve_t_end - retrieve_t_start) * 1000

                    if not self.success:
                        logger.warning(f"Port {self.port}: Frame {self.frame_index}. camera.retrieve() FAILED after successful grab. Retrieve duration: {retrieve_duration_ms:.2f}ms.")


                # Timestamping the frame
                # Using scheduled_current_frame_due_time is the theoretical time this frame was intended for.
                # Using (grab_t_start + retrieve_t_end) / 2 is the actual midpoint of read.
                # Choose one based on your needs for timestamp accuracy vs. consistency.
                # Here, I'm using the actual read time for self.frame_time if successful.
                self.frame_time = (grab_t_start + retrieve_t_end) / 2 if self.success else perf_counter()

                loop_processing_end_time = perf_counter()
                
                # Detailed timing log periodically
                if self.frame_index % log_interval_frames == 0 and self.success:
                    actual_start_of_processing_for_this_frame = current_time_after_potential_sleep # More accurate than scheduled_current_frame_due_time
                    actual_interval = actual_start_of_processing_for_this_frame - getattr(self, '_last_actual_processing_start_time', actual_start_of_processing_for_this_frame)
                    self._last_actual_processing_start_time = actual_start_of_processing_for_this_frame

                    lateness_ms = (current_time_after_potential_sleep - scheduled_current_frame_due_time) * 1000
                    total_read_duration_ms = (retrieve_t_end - grab_t_start) * 1000 if self.success else grab_duration_ms
                    loop_iteration_duration_ms = (loop_processing_end_time - current_time_before_sleep) * 1000


                    logger.debug(f"--- Port {self.port} Frame {self.frame_index} Timing ---")
                    logger.debug(f"  Target Due: {scheduled_current_frame_due_time:.4f}, Actual Start (post-sleep): {current_time_after_potential_sleep:.4f}, Lateness: {lateness_ms:.2f}ms")
                    logger.debug(f"  Actual Interval: {actual_interval*1000:.2f}ms")
                    if grab_success:
                        logger.debug(f"  Grab: {grab_duration_ms:.2f}ms, Retrieve: {retrieve_duration_ms:.2f}ms, Total Read: {total_read_duration_ms:.2f}ms")
                    else:
                        logger.debug(f"  Grab (Failed): {grab_duration_ms:.2f}ms")
                    logger.debug(f"  Frame Timestamp: {self.frame_time:.4f}")
                    logger.debug(f"  Full Loop Iteration (approx): {loop_iteration_duration_ms:.2f}ms")
                    logger.debug(f"  Next Frame Due: {next_frame_due_time:.4f}")
                    logger.debug(f"--------------------------------------")


                if self.success and len(self.subscribers) > 0:
                    if self._show_fps: # Assuming _show_fps is a class member
                        self._add_fps() # Your method to draw FPS on frame

                    self.FPS_actual = self.get_FPS_actual() # Your existing FPS calculation
                    frame_packet = FramePacket(
                        port=self.port,
                        frame_time=self.frame_time,
                        frame_index= self.frame_index,
                        frame=self.frame,
                        fps = self.FPS_actual # Consider if FPS_actual should be smoothed over more frames
                    )

                    for q_idx, q in enumerate(self.subscribers):
                        try:
                            q.put_nowait(frame_packet) # Use put_nowait if subscribers should process fast
                                                    # or q.put() if blocking is acceptable (but can stall this thread)
                        except Exception as e: # queue.Full if using put_nowait and bounded queue
                            logger.warning(f"Port {self.port}: Frame {self.frame_index}. Failed to put FramePacket on subscriber queue {q_idx}. Qsize: {q.qsize()}. Error: {e}")


                self.frame_index +=1
            else: # self.camera.capture is not Opened
                logger.error(f"Port {self.port}: Frame {self.frame_index}. Camera capture is not open. Attempting to re-evaluate next frame time.")
                # If camera is not open, no point in tight loop. Sleep longer.
                sleep(0.1) 
                # Reset next_frame_due_time to avoid rapid catch-up if camera reconnects
                next_frame_due_time = perf_counter() + self.target_interval


        logger.info(f"Port {self.port}: _play_worker thread: Main loop exited. stop_event is set: {self.stop_event.is_set()}")
        self.stop_event.clear() # Clear event for potential reuse if stream is restarted
        self.stop_confirm.put("Successful Stop")
        logger.info(f"Port {self.port}: _play_worker thread: Stop confirmed and event cleared. Thread terminating.")
    
    def _play_worker(self):
        """
        Worker function that is spun up by Thread. Reads in a working frame,
        calls various frame processing methods on it, and updates the exposed
        frame
        """
        self.frame_index = 0
        self.start_time = perf_counter()  # used to get initial delta_t for FPS
        
        # jdw added initialization of next_frame_due_time
        next_frame_due_time = perf_counter() + self.target_interval # Schedule the first frame

        first_time = True
        while not self.stop_event.is_set():
            if first_time:
                logger.info(f"Camera now rolling at port {self.port}")
                first_time = False
            
            #jdw set sleep duration
            current_time = perf_counter()
            sleep_duration = next_frame_due_time - current_time

            if sleep_duration > 0.001:
                sleep(sleep_duration)

            next_frame_due_time += self.target_interval
            if perf_counter() > next_frame_due_time + (self.target_interval * 3): # Tunable factor (3)
                logger.warning(f"Port {self.port}: Significant lag detected. Resetting frame schedule.")
                next_frame_due_time = perf_counter() + self.target_interval

            if self.camera.capture.isOpened():
                # slow wait if not pushing frames
                # this is a sub-optimal busy wait spin lock, but it works and I'm tired.
                # stop_event condition added to allow loop to wrap up
                # if attempting to change resolution
                spinlock_looped = False
                while len(self.subscribers) == 0 and not self.stop_event.is_set():
                    if not spinlock_looped:
                        logger.info(f"Spinlock initiated at port {self.port}")
                        spinlock_looped = True
                    sleep(0.5)
                if spinlock_looped == True:
                    logger.info(f"Spinlock released at port {self.port}")

                # Wait an appropriate amount of time to hit the frame rate target
                sleep(self.wait_to_next_frame())

                read_start = perf_counter()
                grab_success = self.camera.capture.grab()
                self.success, self.frame = self.camera.capture.retrieve()

                read_stop = perf_counter()
                self.frame_time = (read_start + read_stop) / 2

                if self.success and len(self.subscribers) > 0:
                    # logger.info(f"Pushing frame to reel at port {self.port}")

                    if self._show_fps:
                        self._add_fps()

                    # Rate of calling recalc must be frequency of this loop

                    self.FPS_actual = self.get_FPS_actual()
                    frame_packet = FramePacket(
                        port=self.port,
                        frame_time=self.frame_time,
                        frame_index= self.frame_index,
                        frame=self.frame,
                        fps = self.FPS_actual
                    )

                    # cv2.imshow(str(self.port), frame_packet.frame_with_points)
                    # key = cv2.waitKey(1)
                    # if key == ord("q"):
                    #     cv2.destroyAllWindows()
                    #     break

                    for q in self.subscribers:
                        q.put(frame_packet)

                self.frame_index +=1

        logger.info(f"Stream stopped at port {self.port}")
        self.stop_event.clear()
        self.stop_confirm.put("Successful Stop")

    def change_resolution(self, res):
        logger.info(f"About to stop camera at port {self.port}")
        self.stop_event.set()
        self.stop_confirm.get()
        logger.info(f"Roll camera stop confirmed at port {self.port}")

        self.FPS_actual = 0
        self.avg_delta_time = None

        # reconnecting a few times without disconnnect sometimes crashed python
        logger.info(f"Disconnecting from port {self.port}")
        self.camera.disconnect()
        logger.info(f"Reconnecting to port {self.port}")
        self.camera.connect()

        self.camera.size = res
        # Spin up the thread again now that resolution is changed
        logger.info(
            f"Beginning roll_camera thread at port {self.port} with resolution {res}"
        )
        self.thread = Thread(target=self._play_worker, args=(), daemon=True)
        self.thread.start()

    def _add_fps(self):
        """NOTE: this is used in F5 test, not in external use"""
        self.fps_text = str(int(round(self.FPS_actual, 0)))
        cv2.putText(
            self.frame,
            "FPS:" + self.fps_text,
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            3,
        )


if __name__ == "__main__":
    ports = [0]

    cams = []
    for port in ports:
        print(f"Creating camera {port}")
        cam = Camera(port)
        cam.exposure = -7
        cams.append(cam)

    # standard inverted charuco
    frame_packet_queues = {}

    streams = []
    for cam in cams:
        q = Queue(-1)
        frame_packet_queues[cam.port] = q

        print(f"Creating Video Stream for camera {cam.port}")
        stream = LiveStream(cam, fps_target=30)
        stream.subscribe(frame_packet_queues[cam.port])
        stream._show_fps = True
        streams.append(stream)

    while True:
        try:
            for port in ports:
                frame_packet = frame_packet_queues[port].get()

                cv2.imshow(
                    (str(port) + ": 'q' to quit"),
                    frame_packet.frame,
                )

        # bad reads until connection to src established
        except AttributeError:
            pass

        key = cv2.waitKey(1)

        if key == ord("q"):
            for stream in streams:
                stream.camera.capture.release()
            cv2.destroyAllWindows()
            exit(0)

        if key == ord("v"):
            for stream in streams:
                print(f"Attempting to change resolution at port {stream.port}")
                stream.change_resolution((640, 480))

        if key == ord("s"):
            for stream in streams:
                stream.stop()
            cv2.destroyAllWindows()
            exit(0)
