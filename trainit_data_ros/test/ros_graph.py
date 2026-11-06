"""An in-process ROS graph for the acquisition tests: one rclpy context on a private domain, one executor.

Needs a sourced ROS 2 Humble with trainit_hardware_layer >= 0.2.2 (the registered-depth mock). The tests
using it skip cleanly without rclpy.
"""
import os
import threading
import time

DOMAIN_ID = int(os.environ.get('TRAINIT_TEST_ROS_DOMAIN', '87'))     # never run two sessions on one domain


class Graph:
    def __init__(self):
        self._start()

    def _start(self):
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        self.rclpy = rclpy
        self.closed = False
        self.ctx = rclpy.Context()
        rclpy.init(context=self.ctx, domain_id=DOMAIN_ID)
        self.executor = MultiThreadedExecutor(num_threads=16, context=self.ctx)
        self.nodes = []
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        try:
            self.executor.spin()
        except Exception:  # noqa: BLE001 -- stops at shutdown
            pass

    def add(self, node):
        self.nodes.append(node)
        self.executor.add_node(node)
        return node

    def remove(self, node):
        self.executor.remove_node(node)
        if hasattr(node, 'shutdown'):
            node.shutdown()
        node.destroy_node()
        self.nodes.remove(node)

    def helper(self, name='test_helper', use_sim_time=False):
        from rclpy.node import Node
        overrides = self.params({'use_sim_time': True}) if use_sim_time else []
        return self.add(Node(name, context=self.ctx, start_parameter_services=False,
                             parameter_overrides=overrides))

    @staticmethod
    def params(values: dict):
        from rclpy.parameter import Parameter
        return [Parameter(k, value=v) for k, v in values.items()]

    def call(self, node, service: str, timeout=30.0):
        from std_srvs.srv import Trigger
        client = node.create_client(Trigger, service)
        if not client.wait_for_service(timeout_sec=10.0):
            raise AssertionError(f'service {service} not available')
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        node.destroy_client(client)
        assert future.done(), f'{service} did not answer in {timeout} s'
        return future.result()

    def close(self):
        if self.closed:
            return
        self.closed = True
        for node in reversed(list(self.nodes)):
            self.remove(node)
        self.executor.shutdown(timeout_sec=5.0)
        self.rclpy.shutdown(context=self.ctx)
        self._thread.join(timeout=5.0)

    def reset(self):
        """A fresh context and executor (every node of the old graph destroyed). Tests use it between phases
        instead of removing nodes from a spinning executor: under Humble's MultiThreadedExecutor that was seen to
        stop dispatching timers and services afterwards (test-harness flakiness, not product behaviour)."""
        self.close()
        self._start()


def wait_for(predicate, timeout=10.0, step=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


def sim_clock(graph, seconds=10.0, rate_hz=20.0):
    """A /clock that is published but FROZEN at ``seconds`` (a paused simulation)."""
    from rosgraph_msgs.msg import Clock
    node = graph.helper('test_sim_clock')
    pub = node.create_publisher(Clock, '/clock', 10)
    msg = Clock()
    msg.clock.sec = int(seconds)
    node.create_timer(1.0 / rate_hz, lambda: pub.publish(msg))
    return node


def static_tf(graph, parent='base_link', child='camera_color_optical_frame'):
    """What a robot_state_publisher would provide: one static transform."""
    from geometry_msgs.msg import TransformStamped
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
    node = graph.helper('test_static_tf')
    t = TransformStamped()
    t.header.stamp = node.get_clock().now().to_msg()
    t.header.frame_id, t.child_frame_id = parent, child
    t.transform.translation.x, t.transform.translation.z = 0.5, 0.75
    t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = \
        0.7071067811865476, -0.7071067811865476, 0.0, 0.0
    node._broadcaster = StaticTransformBroadcaster(node)
    node._broadcaster.sendTransform(t)
    return node


def fake_camera(graph, *, color=True, depth=True, infos=True, depth_frame=None, stamp_offset_s=0.0, rate_hz=20.0,
                namespace='/camera', frame='camera_color_optical_frame', name='test_fake_camera',
                use_sim_time=False, vary=False, depth_lag_ms=0.0, fx=lambda tick: 8.0):
    """A deliberately non-conforming / partial camera, for the failure suite."""
    import numpy as np
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    node = graph.helper(name, use_sim_time=use_sim_time)
    best = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=5)
    rel = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=5)
    pubs = {k: node.create_publisher(Image, f'{namespace}/{k}', best)
            for k in ('color/image_raw', 'depth/image_rect_raw')}
    infos_pub = {k: node.create_publisher(CameraInfo, f'{namespace}/{k}', rel)
                 for k in ('color/camera_info', 'depth/camera_info')}
    w, h = 16, 12
    rgb = np.full((h, w, 3), 77, np.uint8).tobytes()
    dep = np.full((h, w), 1000, '<u2').tobytes()

    def info(f, stamp, focal):
        ci = CameraInfo(width=w, height=h, distortion_model='plumb_bob', d=[0.0] * 5,
                        k=[focal, 0.0, 8.0, 0.0, focal, 6.0, 0.0, 0.0, 1.0],
                        r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        p=[focal, 0.0, 8.0, 0.0, 0.0, focal, 6.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        ci.header.stamp, ci.header.frame_id = stamp, f
        return ci

    counter = {'tick': 0}

    def tick():
        from rclpy.time import Time
        n = counter['tick']
        counter['tick'] += 1
        now = node.get_clock().now().nanoseconds + int(stamp_offset_s * 1e9)
        stamp = Time(nanoseconds=now).to_msg()
        depth_stamp = Time(nanoseconds=now + int(depth_lag_ms * 1e6)).to_msg()
        colour_bytes = np.full((h, w, 3), n % 251, np.uint8).tobytes() if vary else rgb
        depth_bytes = np.full((h, w), 1000 + n, '<u2').tobytes() if vary else dep
        if color:
            m = Image(height=h, width=w, encoding='rgb8', step=w * 3, data=colour_bytes)
            m.header.stamp, m.header.frame_id = stamp, frame
            pubs['color/image_raw'].publish(m)
        if depth:
            m = Image(height=h, width=w, encoding='16UC1', step=w * 2, data=depth_bytes)
            m.header.stamp, m.header.frame_id = depth_stamp, depth_frame or frame
            pubs['depth/image_rect_raw'].publish(m)
        if infos:
            infos_pub['color/camera_info'].publish(info(frame, stamp, float(fx(n))))
            infos_pub['depth/camera_info'].publish(info(depth_frame or frame, depth_stamp, float(fx(n))))
    node.create_timer(1.0 / rate_hz, tick)
    node.ticks = counter
    return node


def moving_tf(graph, parent='world', child='camera_color_optical_frame', rate_hz=100.0):
    """A moving camera: x(t) = t (seconds, modulo 1000) -- linear, so a lookup AT a stamp is exactly x(stamp)."""
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import TransformBroadcaster
    node = graph.helper('test_moving_tf')
    node._broadcaster = TransformBroadcaster(node)

    def tick():
        now = node.get_clock().now()
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id, t.child_frame_id = parent, child
        t.transform.translation.x = (now.nanoseconds % 1_000_000_000_000) / 1e9
        t.transform.rotation.w = 1.0
        node._broadcaster.sendTransform(t)
    node.create_timer(1.0 / rate_hz, tick)
    return node


class SimCamera:
    """An Isaac-shaped source (the reference cell's documented graph): it OWNS sim time. Every tick advances the
    simulation by one frame period, publishes /clock, then rgb8 + 32FC1 depth (with NaN / +-inf) stamped with that
    SAME sim time (one render tick), plus ONLY the colour CameraInfo -- all in the colour optical frame.
    ``pause()`` stops the ticks (no clock, no frames: a paused simulation); ``reset(sec)`` jumps sim time back."""

    def __init__(self, graph, *, rate_hz=30.0, start_s=10.0, namespace='/camera', frame='camera_color_optical_frame',
                 color_info=True, depth=True, depth_frame=None, depth_encoding='32FC1'):
        import numpy as np
        from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import CameraInfo, Image
        self.node = graph.helper('test_sim_camera')
        best = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        rel = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        self.clock_pub = self.node.create_publisher(Clock, '/clock', 10)
        self.rgb_pub = self.node.create_publisher(Image, f'{namespace}/color/image_raw', best)
        self.depth_pub = self.node.create_publisher(Image, f'{namespace}/depth/image_rect_raw', best)
        self.info_pub = self.node.create_publisher(CameraInfo, f'{namespace}/color/camera_info', rel)
        self.Clock, self.Image, self.CameraInfo, self.np = Clock, Image, CameraInfo, np
        self.period_ns = int(1e9 / rate_hz)
        self.sim_ns = int(start_s * 1e9)
        self.running = True
        self.frames = 0
        self.frame, self.depth_frame, self.color_info, self.depth = frame, depth_frame or frame, color_info, depth
        self.depth_encoding = depth_encoding
        self.w, self.h = 16, 12
        self.node.create_timer(1.0 / rate_hz, self._tick)

    def pause(self):
        self.running = False

    def resume(self):
        self.running = True

    def reset(self, seconds=0.5):
        self.sim_ns = int(seconds * 1e9)

    def depth_array(self, n):
        np = self.np
        depth = np.full((self.h, self.w), 0.5 + (n % 100) / 1000.0, np.float32)
        depth[0, 0], depth[0, 1], depth[0, 2] = np.nan, np.inf, -np.inf
        return depth

    def _tick(self):
        if not self.running:
            return
        self.sim_ns += self.period_ns
        sec, nanosec = divmod(self.sim_ns, 1_000_000_000)
        clock = self.Clock()
        clock.clock.sec, clock.clock.nanosec = sec, nanosec
        self.clock_pub.publish(clock)
        n = self.frames
        self.frames += 1
        rgb = self.Image(height=self.h, width=self.w, encoding='rgb8', step=self.w * 3,
                         data=self.np.full((self.h, self.w, 3), n % 251, self.np.uint8).tobytes())
        rgb.header.stamp.sec, rgb.header.stamp.nanosec, rgb.header.frame_id = sec, nanosec, self.frame
        self.rgb_pub.publish(rgb)
        if self.depth:
            if self.depth_encoding == '32FC1':
                data, step = self.depth_array(n).astype('<f4').tobytes(), self.w * 4
            else:
                data, step = bytes(self.w * self.h), self.w
            depth = self.Image(height=self.h, width=self.w, encoding=self.depth_encoding, step=step, data=data)
            depth.header.stamp.sec, depth.header.stamp.nanosec, depth.header.frame_id = sec, nanosec, self.depth_frame
            self.depth_pub.publish(depth)
        if self.color_info:
            info = self.CameraInfo(width=self.w, height=self.h, distortion_model='plumb_bob', d=[0.0] * 5,
                                   k=[8.0, 0.0, 8.0, 0.0, 8.0, 6.0, 0.0, 0.0, 1.0],
                                   r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                                   p=[8.0, 0.0, 8.0, 0.0, 0.0, 8.0, 6.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            info.header.stamp.sec, info.header.stamp.nanosec, info.header.frame_id = sec, nanosec, self.frame
            self.info_pub.publish(info)
