from math import pi
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import tf2_py
from tf2_ros import TransformBroadcaster, TransformListener, TransformStamped
import random
import tf2_ros
import ros2param

class PositionUpdater:
    def __init__(self, step) -> None:
        self.targets = []
        self.step = step

    def updatePositions(self, jointPositions):
        newJointPositions = [self.updatePosition(joint, target) for joint, target in zip(jointPositions, self.targets)]
        return newJointPositions

    def updateTarget(self, targets):
        self.targets = targets

    def updatePosition(self, joint, target):
        if target > joint:
            return joint + self.step
        if target < joint:
            return joint - self.step
        return joint

    def isAtTarget(self, jointPositions):
        def isNear(target, joint):
            if abs(target - joint) < 2 * self.step:
                return True

        isAt = [isNear(target, joint) for target, joint in zip(self.targets, jointPositions)]

        return all(isAt)

class StatePublisher(Node):

    def __init__(self):
        rclpy.init()
        super().__init__('state_publisher')

        qos_profile = QoSProfile(depth=10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', qos_profile)  # JointStates
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.nodeName = self.get_name()
        self.get_logger().info("{0} started".format(self.nodeName))

        degree = pi / 180.0
        loop_rate = self.create_rate(30)

        # robot state
        joints = [0., 0., 0., 0., 0.]

        joint_state = JointState()
        positionUpdater = PositionUpdater(step=degree)
        positionUpdater.updateTarget([1., 1.5, 1.2, 1.3, .7])


        try:
            while rclpy.ok():
                rclpy.spin_once(self)

                # update joint_state
                now = self.get_clock().now()
                joint_state.header.stamp = now.to_msg()
                joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
                joint_state.position = joints

                # send the joint state and transform
                self.joint_pub.publish(joint_state)

                joints = positionUpdater.updatePositions(joints)
                if positionUpdater.isAtTarget(joints):
                    positionUpdater.updateTarget([random.random() * 2 for _ in joints])


                loop_rate.sleep()

        except KeyboardInterrupt:
            pass


def main():
    node = StatePublisher()


if __name__ == '__main__':
    main()
