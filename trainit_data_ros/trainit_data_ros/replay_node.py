"""The replay publisher: a SEALED session back onto the SAME ROS camera contract a live source publishes.

A replay is one more live source (ADR-0029 Decision 4): the recorder records it like any camera, with
``backend: replay`` and ``replay_of: <session id>`` (its recording configuration says so), and inherits
the original session's source descriptor.

What is preserved and what is not:
- payload VALUES exactly: each image is published with its stored encoding (rgb8 / 16UC1 / 32FC1), so the
  bytes a recorder captures again are the original arrays bit for bit;
- the CameraInfo: rebuilt from the observation's CalibrationRecord (K, D, R, P, binning, roi verbatim);
- the extrinsics: each calibration's extrinsics are re-published on /tf_static, so a re-recording sees
  the transforms the original runtime believed;
- stamps are NOT preserved: every capture is re-stamped with this node's clock at publish time (the
  original intra-capture offsets between streams are kept). The original stamps stay in the original
  session; the re-recording's descriptor declares the re-stamping.

Topics: the original session's recorded image topics, each with the contract's CameraInfo beside it, unless
``camera_namespace`` re-roots them onto the contract names under another namespace. The replay is therefore a
fully conforming source (it publishes depth/camera_info even when the original source did not).

Limits (refused or documented, never silent): a session whose extrinsic changed between captures is refused
(one /tf_static per frame pair cannot replay it); a moving camera's ``pose_at_stamp`` transforms are not
re-published, so re-recording a moving-camera session reports TF_UNAVAILABLE for its pose.

Parameters: ``session_id`` (required), ``store_root`` (default ``$TRAINIT_DATA_ROOT`` / ``~/TrainItData``),
``camera_namespace`` (optional re-rooting), ``rate_hz`` (captures per second, default 5), ``loop`` (default
true). Only COMPLETE / PARTIAL captures' observations are published.
"""
import threading
import time

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, RegionOfInterest
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from trainit_data.store import Store

from .contract import CAMERA_STREAMS, join

SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=5)
INFO_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=5)
_CHANNELS = {'rgb8': 3, '16UC1': 1, '32FC1': 1, 'mono8': 1, 'mono16': 1}


def image_msg(array, encoding: str) -> Image:
    msg = Image(height=int(array.shape[0]), width=int(array.shape[1]), encoding=encoding, is_bigendian=0)
    little = array.astype(array.dtype.newbyteorder('<'), copy=False)
    msg.step = int(array.shape[1]) * _CHANNELS[encoding] * array.dtype.itemsize
    msg.data = little.tobytes()
    return msg


def info_msg(record) -> CameraInfo:
    i = record.intrinsics
    msg = CameraInfo(width=i.width, height=i.height, distortion_model=i.distortion_model, d=list(i.D), k=list(i.K),
                     r=list(i.R), p=list(i.P), binning_x=i.binning[0], binning_y=i.binning[1])
    msg.roi = RegionOfInterest(x_offset=i.roi['x'], y_offset=i.roi['y'], width=i.roi['width'],
                               height=i.roi['height'], do_rectify=i.roi['do_rectify'])
    return msg


class Replay(Node):
    def __init__(self, **node_kwargs):
        super().__init__('trainit_data_replay', **node_kwargs)
        self.declare_parameter('session_id', '')
        self.declare_parameter('store_root', '')
        self.declare_parameter('camera_namespace', '')
        self.declare_parameter('rate_hz', 5.0)
        self.declare_parameter('loop', True)
        store = Store.open(self.get_parameter('store_root').value or None)
        self.reader = store.open_session(self.get_parameter('session_id').value)
        if not self.reader.header.sealed:
            raise RuntimeError(f'session {self.reader.session_id} is {self.reader.header.status}: replay reads '
                               'sealed sessions only')
        namespace = self.get_parameter('camera_namespace').value
        self.loop = bool(self.get_parameter('loop').value)
        self.captures = [c for c in self.reader.captures() if c.observations]     # the index, streamed once
        self.calibrations = {}
        transforms = {}
        for capture in self.captures:
            for obs in capture.observations:
                cal_id = obs.data.calibration_id
                if cal_id and cal_id not in self.calibrations:
                    record = store.get_calibration(cal_id, self.reader.vocab)
                    self.calibrations[cal_id] = record
                    for ext in record.extrinsics:
                        pair = (ext.parent_frame, ext.child_frame)
                        if pair in transforms and transforms[pair] != ext:
                            raise RuntimeError(f'session {self.reader.session_id}: the extrinsic {pair} changed '
                                               'between captures; one /tf_static cannot replay it')
                        transforms[pair] = ext
        self.pubs = {}
        self.skipped = []           # declared streams that are not camera images (e.g. a ``vector`` IMU reading)
        for sensor in self.reader.header.sensors:
            for decl in sensor.streams:
                if self.reader.vocab.modalities[decl.modality]['data_block'] != 'image':
                    self.skipped.append(f'{sensor.sensor_id}/{decl.stream} ({decl.modality})')
                    continue        # replay re-publishes the camera contract only
                topic, info = self._topics(decl, namespace)
                self.pubs[(sensor.sensor_id, decl.stream)] = (
                    self.create_publisher(Image, topic, SENSOR_QOS), self.create_publisher(CameraInfo, info, INFO_QOS))
        self._static = StaticTransformBroadcaster(self)
        if transforms:
            self._static.sendTransform([self._transform(e) for e in transforms.values()])
        self.index = 0
        self.published = 0
        self.errors = 0
        # A dedicated publishing thread, not an rclpy timer: under Humble's multi-threaded executor a timer was
        # seen to stop after its first tick (and to fire twice in the recorder). Publishing needs no executor.
        self._period_s = 1.0 / float(self.get_parameter('rate_hz').value)
        self._running = True
        self._thread = threading.Thread(target=self._run, name='trainit_data_replay', daemon=True)
        self._thread.start()
        onto = f'the namespace {namespace}' if namespace else 'its recorded topics'
        self.get_logger().info(f'replaying {self.reader.session_id}: {len(self.captures)} capture(s) onto {onto}, '
                               're-stamped at publish')
        if self.skipped:
            self.get_logger().warning('not replayed (not a camera image stream): ' + ', '.join(self.skipped))

    @staticmethod
    def _topics(decl, namespace):
        """The image topic the original recorded, and the contract's CameraInfo name beside it (each stream
        publishes its own CameraInfo, registered depth an identical one: contracts/camera.md)."""
        contract_topic, contract_info = CAMERA_STREAMS[decl.stream][:2]
        if namespace or decl.topic is None:
            return join(namespace or '/camera', contract_topic), join(namespace or '/camera', contract_info)
        root = decl.topic[:-len(contract_topic)] if decl.topic.endswith(contract_topic) else None
        if root is None:
            raise RuntimeError(f'{decl.topic!r} is not the contract name of a {decl.stream} stream: set '
                               'camera_namespace')
        return decl.topic, root + contract_info

    def _transform(self, ext) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = ext.parent_frame, ext.child_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = ext.translation
        r = t.transform.rotation
        r.x, r.y, r.z, r.w = ext.rotation_xyzw
        return t

    def _run(self):
        next_tick = time.monotonic()
        while self._running:
            self._tick()
            next_tick += self._period_s
            time.sleep(max(0.0, next_tick - time.monotonic()))

    def shutdown(self):
        self._running = False
        self._thread.join(timeout=5.0)

    def _tick(self):
        try:
            self._publish_next()
        except Exception as exc:  # noqa: BLE001 -- one unreadable payload must not stop the node
            self.errors += 1
            self.get_logger().error(f'replay: {exc!r}')

    def _publish_next(self):
        if not self.captures or (self.index >= len(self.captures) and not self.loop):
            return
        capture = self.captures[self.index % len(self.captures)]
        self.index += 1
        now = self.get_clock().now().nanoseconds
        first = min(o.stamp.to_ns() for o in capture.observations)     # offsets stay those of the whole capture
        for obs in capture.observations:
            if (obs.sensor_id, obs.stream) not in self.pubs:
                continue
            stamp_ns = now + (obs.stamp.to_ns() - first)            # original offsets kept
            image = image_msg(self.reader.load_payload(obs), obs.data.encoding)
            image.header.frame_id = obs.data.frame_id
            image.header.stamp.sec, image.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
            pub_image, pub_info = self.pubs[(obs.sensor_id, obs.stream)]
            if obs.data.calibration_id in self.calibrations:
                info = info_msg(self.calibrations[obs.data.calibration_id])
                info.header = image.header
                pub_info.publish(info)
            pub_image.publish(image)
        self.published += 1


def main(args=None):
    rclpy.init(args=args)
    node = Replay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
