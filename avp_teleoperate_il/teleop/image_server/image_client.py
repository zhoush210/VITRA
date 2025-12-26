import argparse
import os
from pathlib import Path
import struct
import time
from collections import deque
from multiprocessing import shared_memory

import cv2
import numpy as np
import zmq

try:
    import logging_mp
    logger_mp = logging_mp.get_logger(__name__)
    logging_mp.basic_config(level=logging_mp.INFO)
except Exception:  # fallback to std logging if custom logger missing
    import logging
    logging.basicConfig(level=logging.INFO)
    logger_mp = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="192.168.123.164", help="ip address for the server")
parser.add_argument("--save-dir", default=None, help="optional directory to save received images (creates head/ wrist subfolders)")
parser.add_argument("--head-cams", type=int, default=1, help="number of cameras concatenated in the head image (default: 1)")
parser.add_argument("--wrist-cams", type=int, default=2, help="number of wrist cameras concatenated in the wrist image (default: 0)")


class ImageClient:
    """Client that supports bino head cams + two wrist cams, time-sync handshake, optional saving, and shared memory output."""

    def __init__(
        self,
        tv_img_shape=None,
        tv_img_shm_name=None,
        wrist_img_shape=None,
        wrist_img_shm_name=None,
        delay_shm_name=None,
        image_show=False,
        server_address="192.168.123.164",
        port=5555,
        handshake_port=5556,
        save_dir=None,
        head_cam_count=1,
        wrist_cam_count=2,
    ):
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._port = port
        self._handshake_port = handshake_port

        self.tv_img_shape = tv_img_shape
        self.wrist_img_shape = wrist_img_shape

        # Saving options
        self.save_dir = save_dir
        self.head_cam_count = int(head_cam_count)
        self.wrist_cam_count = int(wrist_cam_count)
        self.save_dir_path = None
        if self.save_dir:
            self.save_dir_path = Path(self.save_dir)
            try:
                (self.save_dir_path / "head").mkdir(parents=True, exist_ok=True)
                (self.save_dir_path / "wrist" / "left").mkdir(parents=True, exist_ok=True)
                (self.save_dir_path / "wrist" / "right").mkdir(parents=True, exist_ok=True)
                logger_mp.info(f"ImageClient will save frames into: {self.save_dir_path}")
            except Exception as e:
                logger_mp.error(f"Failed to create save directories: {e}")

        # --- ZMQ and Time Sync Initialization ---
        self._context = zmq.Context()
        self._socket = None
        self.time_offset = 0

        # --- Shared Memory Setup ---
        self.tv_enable_shm = self._setup_shm(tv_img_shm_name, tv_img_shape, "tv")
        self.wrist_enable_shm = self._setup_shm(wrist_img_shm_name, wrist_img_shape, "wrist")

        self.delay_enable_shm = False
        if delay_shm_name is not None:
            try:
                self.delay_shm = shared_memory.SharedMemory(name=delay_shm_name)
                self.delay_array = np.ndarray((), dtype=np.float64, buffer=self.delay_shm.buf)
                self.delay_enable_shm = True
                logger_mp.info(f"Successfully attached to shared memory '{delay_shm_name}' for delay.")
            except FileNotFoundError:
                logger_mp.error(f"Shared memory block '{delay_shm_name}' not found. Cannot share delay.")

        # --- Moving average logging ---
        self._delay_history = deque(maxlen=500)
        self._last_log_time = time.time()
        self._log_interval_sec = 10.0

    def _setup_shm(self, shm_name, img_shape, prefix):
        if img_shape is not None and shm_name is not None:
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                setattr(self, f"{prefix}_image_shm", shm)
                setattr(self, f"{prefix}_img_array", np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf))
                logger_mp.info(f"Successfully attached to shared memory '{shm_name}'.")
                return True
            except FileNotFoundError:
                logger_mp.error(f"Shared memory block '{shm_name}' not found. Please ensure the consumer process has created it.")
                return False
        return False

    def _perform_handshake(self, num_samples=100):
        logger_mp.info(f"Performing handshake with server at {self._server_address}:{self._handshake_port}...")
        handshake_socket = self._context.socket(zmq.REQ)
        handshake_socket.connect(f"tcp://{self._server_address}:{self._handshake_port}")

        offsets, rtts = [], []
        try:
            handshake_socket.setsockopt(zmq.RCVTIMEO, 2000)
            handshake_socket.setsockopt(zmq.LINGER, 0)

            for i in range(num_samples):
                t0 = time.time()
                handshake_socket.send(b"sync")
                try:
                    packed_server_time = handshake_socket.recv()
                    t1 = time.time()
                    server_time = struct.unpack("d", packed_server_time)[0]
                    rtt = t1 - t0
                    offset = server_time - (t0 + rtt / 2)
                    rtts.append(rtt)
                    offsets.append(offset)
                    time.sleep(0.05)
                except zmq.Again:
                    logger_mp.warning(f"Handshake sample {i+1}/{num_samples} failed: No response.")
                    continue
                except Exception as e:
                    logger_mp.error(f"Error during handshake sample {i+1}/{num_samples}: {e}")
                    continue

            if not offsets:
                logger_mp.error("Handshake failed: No successful samples obtained.")
                return False

            self.time_offset = sum(offsets) / len(offsets)
            average_rtt = sum(rtts) / len(rtts)
            logger_mp.info(f"Handshake successful after {len(offsets)} samples!")
            logger_mp.info(f"  - Average Network RTT: {average_rtt * 1000:.2f} ms")
            logger_mp.info(f"  - Estimated clock offset: {self.time_offset:.4f} s (Server ahead if > 0)")

        except Exception as e:
            logger_mp.error(f"An error occurred during handshake: {e}")
            return False
        finally:
            handshake_socket.close()

        return True

    def _close(self):
        if self._socket:
            self._socket.close()
        self._context.term()
        if self._image_show:
            cv2.destroyAllWindows()
        logger_mp.info("Image client has been closed.")

    def _write_delay(self, delay):
        if not self.delay_enable_shm:
            return
        try:
            self.delay_array[()] = delay
        except Exception as e:
            logger_mp.warning(f"Failed to write delay to shared memory: {e}")

    def _log_delay(self):
        current_time = time.time()
        if current_time - self._last_log_time >= self._log_interval_sec:
            if self._delay_history:
                avg_delay = sum(self._delay_history) / len(self._delay_history)
                logger_mp.info(
                    f"Average Frame Latency (last {len(self._delay_history)} frames): {avg_delay * 1000:.2f} ms"
                )
            self._last_log_time = current_time

    def _save_images(self, topic, img, adjusted_server_time, frame_count):
        if not self.save_dir_path:
            return
        ts_ms = int(adjusted_server_time * 1000)
        try:
            if topic == "combined":
                # Save head cams
                if self.tv_img_shape is not None and self.head_cam_count > 0:
                    expected_w = int(self.tv_img_shape[1])
                    if img.shape[1] >= expected_w:
                        head_region = img[:, :expected_w]
                        each_w = max(1, head_region.shape[1] // self.head_cam_count)
                        for i in range(self.head_cam_count):
                            x0 = i * each_w
                            x1 = (i + 1) * each_w if i < self.head_cam_count - 1 else head_region.shape[1]
                            sub = head_region[:, x0:x1]
                            fname = self.save_dir_path / "head" / f"head_cam{i}_{frame_count:06d}_{ts_ms}.jpg"
                            cv2.imwrite(str(fname), sub)
                    else:
                        fname = self.save_dir_path / "head" / f"head_full_{frame_count:06d}_{ts_ms}.jpg"
                        cv2.imwrite(str(fname), img)

                # Extract wrist regions from combined if possible
                if self.wrist_img_shape is not None and self.tv_img_shape is not None:
                    head_w = int(self.tv_img_shape[1])
                    single_w = int(self.wrist_img_shape[1]) if self.wrist_cam_count == 1 else int(self.wrist_img_shape[1] // 2)

                    left_region = right_region = rem = None
                    if img.shape[1] >= head_w + single_w:
                        left_region = img[:, head_w : head_w + single_w]
                        fname_l = self.save_dir_path / "wrist" / "left" / f"wrist_left_{ts_ms}.jpg"
                        cv2.imwrite(str(fname_l), left_region)
                    if img.shape[1] >= head_w + 2 * single_w and self.wrist_cam_count >= 2:
                        right_region = img[:, head_w + single_w : head_w + 2 * single_w]
                        fname_r = self.save_dir_path / "wrist" / "right" / f"wrist_right_{ts_ms}.jpg"
                        cv2.imwrite(str(fname_r), right_region)
                    if self.wrist_cam_count == 1 and img.shape[1] > head_w:
                        # 仅取最右侧一个腕部宽度，避免把多余的头/腕拼接段写入
                        start = max(head_w, img.shape[1] - self.wrist_img_shape[1])
                        rem = img[:, start : start + self.wrist_img_shape[1]]
                        fname_r = self.save_dir_path / "wrist" / "right" / f"wrist_right_{ts_ms}.jpg"
                        cv2.imwrite(str(fname_r), rem)

            elif topic == "wrist_left":
                fname = self.save_dir_path / "wrist" / "left" / f"wrist_left_{ts_ms}.jpg"
                cv2.imwrite(str(fname), img)

            elif topic == "wrist_right":
                fname = self.save_dir_path / "wrist" / "right" / f"wrist_right_{ts_ms}.jpg"
                cv2.imwrite(str(fname), img)

        except Exception as e:
            logger_mp.warning(f"Failed to save images: {e}")

    def _write_shared_memory(self, topic, img):
        # Head cameras from combined
        if topic == "combined" and self.tv_enable_shm and self.tv_img_shape is not None:
            if img.shape[1] >= self.tv_img_shape[1]:
                np.copyto(self.tv_img_array, img[:, : self.tv_img_shape[1]])
            else:
                logger_mp.warning("[Image Client] TV image part is smaller than expected shape.")

        # Wrist extraction from combined
        if topic == "combined" and self.wrist_enable_shm and self.wrist_img_shape is not None and self.tv_img_shape is not None:
            try:
                head_w = int(self.tv_img_shape[1])
                if self.wrist_cam_count == 2:
                    single_w = int(self.wrist_img_shape[1] // 2)
                    if img.shape[1] >= head_w + 2 * single_w:
                        left_wrist = img[:, head_w : head_w + single_w]
                        right_wrist = img[:, head_w + single_w : head_w + 2 * single_w]
                        combined_wrist = np.concatenate([left_wrist, right_wrist], axis=1)
                        if combined_wrist.shape[0] == self.wrist_img_shape[0] and combined_wrist.shape[1] == self.wrist_img_shape[1]:
                            np.copyto(self.wrist_img_array, combined_wrist)
                        else:
                            resized = cv2.resize(combined_wrist, (self.wrist_img_shape[1], self.wrist_img_shape[0]))
                            np.copyto(self.wrist_img_array, resized)
                elif self.wrist_cam_count == 1:
                    if img.shape[1] > head_w:
                        # 仅取最右侧一个腕部宽度，防止包含额外的头/腕拼接段
                        start = max(head_w, img.shape[1] - self.wrist_img_shape[1])
                        wrist_region = img[:, start : start + self.wrist_img_shape[1]]
                        if wrist_region.shape[0] == self.wrist_img_shape[0] and wrist_region.shape[1] == self.wrist_img_shape[1]:
                            np.copyto(self.wrist_img_array, wrist_region)
                        else:
                            resized = cv2.resize(wrist_region, (self.wrist_img_shape[1], self.wrist_img_shape[0]))
                            np.copyto(self.wrist_img_array, resized)
            except Exception as e:
                logger_mp.warning(f"[Image Client] Failed to extract wrist from combined image: {e}")

        # Wrist-right topic to shared memory
        if topic == "wrist_right" and self.wrist_enable_shm and self.wrist_img_shape is not None:
            try:
                if img.shape[1] >= self.wrist_img_shape[1] and img.shape[0] >= self.wrist_img_shape[0]:
                    np.copyto(self.wrist_img_array, img)
                else:
                    resized = cv2.resize(img, (self.wrist_img_shape[1], self.wrist_img_shape[0]))
                    np.copyto(self.wrist_img_array, resized)
            except Exception:
                try:
                    resized = cv2.resize(img, (self.wrist_img_shape[1], self.wrist_img_shape[0]))
                    np.copyto(self.wrist_img_array, resized)
                except Exception:
                    logger_mp.warning("[Image Client] Wrist image part is smaller than expected shape and resize failed.")

    def receive_process(self):
        # 1) Handshake
        if not self._perform_handshake():
            self._close()
            return

        # 2) Connect to stream
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{self._server_address}:{self._port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        logger_mp.info("Connected to image stream. Waiting to receive data...")

        try:
            start_time = time.time()
            frame_count = 0
            while self.running:
                parts = self._socket.recv_multipart()
                receive_time = time.time()

                if len(parts) == 1:
                    topic = "combined"
                    payload = parts[0]
                else:
                    topic = parts[0].decode()
                    payload = parts[1]

                server_timestamp = struct.unpack("d", payload[:8])[0]
                jpg_bytes = payload[8:]

                adjusted_server_time = server_timestamp - self.time_offset
                delay = receive_time - adjusted_server_time
                self._delay_history.append(delay)
                self._log_delay()

                np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                if img is None:
                    logger_mp.warning("[Image Client] Failed to decode image.")
                    continue

                self._save_images(topic, img, adjusted_server_time, frame_count)
                self._write_shared_memory(topic, img)
                self._write_delay(delay)

                if self._image_show:
                    height, width = img.shape[:2]
                    display_width = 800
                    scale = display_width / width
                    resized_image = cv2.resize(img, (display_width, int(height * scale)))
                    delay_text = f"Delay: {delay * 1000:.2f} ms"
                    cv2.putText(resized_image, delay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                    frame_count += 1
                    fps = frame_count / float(time.time() - start_time)
                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(resized_image, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Image Client Stream", resized_image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False

        except KeyboardInterrupt:
            logger_mp.info("Image client interrupted by user.")
        except Exception as e:
            logger_mp.warning(f"[Image Client] An error occurred while receiving data: {e}")
        finally:
            self._close()


if __name__ == "__main__":
    # For quick testing without shared memory or splits
    args = parser.parse_args()
    client = ImageClient(
        image_show=True,
        server_address=args.ip,
        save_dir=args.save_dir,
        head_cam_count=args.head_cams,
        wrist_cam_count=args.wrist_cams,
    )
    client.receive_process()