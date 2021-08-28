from time import sleep
from tg_interfaces.srv import JointExample
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from tf2_ros import TransformListener, TransformStamped, TransformBroadcaster
import tf2_ros
from rclpy.time import Time
import rclpy
from rclpy.node import Node

class JointService(Node):
    """
    Nó composto por:
    Service que dado ângulos das juntas retorna x, y, z, qx, qy, qz, qw
    Publisher que publica juntas para robot_state_publisher
    Listener que escuta posição e orientação de tf
    """
    def __init__(self):
        super().__init__('joint_service')
        self.nodeName = self.get_name()
        self.get_logger().info("{0} started".format(self.nodeName))
        
        qos_profile = QoSProfile(depth=10)

        #service
        self.srv = self.create_service(JointExample, 'joint_example', self.joint_example_callback)
        #publisher
        self.publisher = self.create_publisher(JointState, 'joint_states', qos_profile=qos_profile)
        #listener
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = TransformListener(self.tfBuffer, self)
        
        self.joint_state = JointState()
        self.joints = [0., 0., 0., 0., 0.]
        
        self.previousTrans: TransformStamped = TransformStamped()
        self.newTrans: TransformStamped = TransformStamped()
        self.hasUpdated = False
        asd = True

        try:
            while rclpy.ok():
                rclpy.spin_once(self)
                self.publish_joints()
                if asd:
                    try:
                        self.previousTrans: TransformStamped = self.tfBuffer.lookup_transform('link1', 'gripper', Time())
                        # self.newTrans: TransformStamped = self.tfBuffer.lookup_transform('link1', 'gripper', Time())
                        asd = False
                    except:
                        pass
                # try:
                #     trans: TransformStamped = self.tfBuffer.lookup_transform('link1', 'gripper', Time())
                #     print(trans.transform.translation)
                #     print(trans.transform.rotation)
                #     print('----------')
                # except:
                #     continue
        except KeyboardInterrupt:
            pass

    def updateTransformStatus(self):
        if not self.previousTrans.transform == self.newTrans.transform:
            self.hasUpdated = True
            self.previousTrans = self.newTrans
    
    def publish_joints(self):
        # atualiza joint_state
        now = self.get_clock().now()
        self.joint_state.header.stamp = now.to_msg()
        self.joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
        self.joint_state.position = self.joints

        #publica mensagem
        self.publisher.publish(self.joint_state)

    def joint_example_callback(self, req, res):
        self.joints = [req.joint1, req.joint2, req.joint3, req.joint4, req.joint5]
        self.publish_joints()
        
        # while not self.hasUpdated:
        #     self.newTrans = self.tfBuffer.lookup_transform('link1', 'gripper', Time())

        # self.newTrans = self.tfBuffer.lookup_transform('link1', 'gripper', Time())
        # while not self.hasTransformUpdated(self.previousTrans, self.newTrans):
        #     self.newTrans = self.tfBuffer.lookup_transform('link1', 'gripper', Time())
        # self.previousTrans = self.newTrans

        # self.get_logger().info(f'{req.joint1};{req.joint2}')
        res.x  = self.newTrans.transform.translation.x
        res.y  = self.newTrans.transform.translation.y
        res.z  = self.newTrans.transform.translation.z
        res.qx = self.newTrans.transform.rotation.x
        res.qy = self.newTrans.transform.rotation.y
        res.qz = self.newTrans.transform.rotation.z
        res.qw = self.newTrans.transform.rotation.w
        return res
        
    def hasTransformUpdated(self, previousTrans: TransformStamped, newTrans: TransformStamped):
        print(previousTrans.transform)
        print(newTrans.transform)
        print('--------------')
        if (previousTrans.transform.translation.x == newTrans.transform.translation.x):
            return False
        return True

def main():
    rclpy.init()
    joint_service = JointService()
    # rclpy.spin(joint_service)
    # rclpy.shutdown()

if __name__ == '__main__':
    main()