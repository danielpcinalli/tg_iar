from hashlib import new
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
from rclpy.time import Time
from csv import writer

class StatePublisher(Node):

    def __init__(self):
        rclpy.init()
        super().__init__('training_examples_creator')

        qos_profile = QoSProfile(depth=10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', qos_profile)  # JointStates
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.nodeName = self.get_name()
        self.get_logger().info("{0} started".format(self.nodeName))

        # robot state
        joints = [0., 0., 0., 0., 0.]

        joint_state = JointState()

        self.previousTF: TransformStamped = TransformStamped()
        self.newTF: TransformStamped = TransformStamped()

        #listener
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = TransformListener(self.tfBuffer, self)

        try:

            while rclpy.ok():
                rclpy.spin_once(self)
                
                # update joint_state
                now = self.get_clock().now()
                joint_state.header.stamp = now.to_msg()
                joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
                joint_state.position = joints

                # send the joint state
                self.joint_pub.publish(joint_state)

                try:
                    self.newTF = self.tfBuffer.lookup_transform('link1', 'gripper', Time())
                except:
                    pass

                #caso as posições tenham atualizado, salva e cria novas juntas aleatórias
                if self.hasTransformUpdated():
                    self.previousTF = self.newTF
                    with open('/home/osboxes/results/data.csv', 'a+', newline='') as f:
                        csv_writer = writer(f)
                        newRow = joints + [
                            self.newTF.transform.translation.x,
                            self.newTF.transform.translation.y,
                            self.newTF.transform.translation.z,
                            self.newTF.transform.rotation.x,
                            self.newTF.transform.rotation.y,
                            self.newTF.transform.rotation.z,
                            self.newTF.transform.rotation.w
                        ]
                        csv_writer.writerow(newRow)
                    joints = self.newRandomJoints()
                    
        except KeyboardInterrupt:
            pass

    def newRandomJoints(self):
        newJoints = [
            self.randomBetween(-2*pi, 2*pi),  #livre
            self.randomBetween(-2., 2.),      #-2 a +2
            self.randomBetween(-2., 2.),      #-2 a +2
            self.randomBetween(-2*pi, 2*pi),  #livre
            self.randomBetween(-2., 2.)       #-2 a +2
        ]
        return newJoints

    def randomBetween(self, a, b):
        r = random.random()
        r = r * (b - a)
        r = r + a
        return r

    def hasTransformUpdated(self):
        if (self.previousTF.transform == self.newTF.transform):
            return False
        return True

def main():
    node = StatePublisher()


if __name__ == '__main__':
    main()
