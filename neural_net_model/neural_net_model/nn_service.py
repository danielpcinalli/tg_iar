from time import sleep
from tg_interfaces.srv import ReverseKinematic
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from tf2_ros import TransformListener, TransformStamped, TransformBroadcaster
import tf2_ros
from rclpy.time import Time
import rclpy
from rclpy.node import Node
import pickle
import os

from ament_index_python.packages import get_package_share_directory
class NN_service(Node):
    """
    
    """
    #TODO: carregar nn e atualizar callback para usar nn para obter joints
    def __init__(self):
        # rclpy.init()
        super().__init__('nn_service')
        self.nodeName = self.get_name()
        self.get_logger().info("{0} started".format(self.nodeName))
        
        qos_profile = QoSProfile(depth=10)

        #publisher
        self.joint_pub = self.create_publisher(JointState, 'joint_states', qos_profile)

        #service
        self.srv = self.create_service(ReverseKinematic, 'reverse_kinematic', self.reverse_kinematic_callback)
        nn_model = os.path.join(
                get_package_share_directory('neural_net_model'),
                'neural_model.pkl')
        with open(nn_model, 'rb') as pkl_file:
            self.nn = pickle.load(pkl_file)
        
        self.joints = [0., 0., 0., 0., 0.]
        joint_state = JointState()
        
        try:
            while rclpy.ok():
                rclpy.spin_once(self)
                # update joint_state
                now = self.get_clock().now()
                joint_state.header.stamp = now.to_msg()
                joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
                joint_state.position = self.joints

                # send the joint state
                self.joint_pub.publish(joint_state)
        except Exception as e:
            print(e)            
    

    def reverse_kinematic_callback(self, req, res):

        
        x = req.x
        y = req.y
        z = req.z
        qx = req.qx
        qy = req.qy
        qz = req.qz
        qw = req.qw
        X = [[x, y, z, qx, qy, qz, qw]]
        Y = self.nn.predict(X)[0]
        
        #obter a partir da nn
        res.joint1 = Y[0]
        res.joint2 = Y[1]
        res.joint3 = Y[2]
        res.joint4 = Y[3]
        res.joint5 = Y[4]
        self.joints = [Y[0], Y[1], Y[2], Y[3], Y[4]]
        return res
        

def main():
    rclpy.init()
    service = NN_service()
    # rclpy.spin(service)
    # rclpy.shutdown()

if __name__ == '__main__':
    main()