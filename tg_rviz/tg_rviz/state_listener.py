from math import pi
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import TransformListener, TransformStamped
import tf2_ros

class StateListener(Node):

    def __init__(self):
        rclpy.init()
        super().__init__('state_listener')

        
        self.nodeName = self.get_name()
        self.get_logger().info("{0} started".format(self.nodeName))

        tfBuffer = tf2_ros.Buffer()

        self.listener = TransformListener(tfBuffer, self)
        
        try:
            while rclpy.ok():
                rclpy.spin_once(self)

        
                try:
                    
                    trans: TransformStamped = tfBuffer.lookup_transform('link1', 'link6', Time())
                    x, y, z = (trans.transform.translation.x, 
                               trans.transform.translation.y, 
                               trans.transform.translation.z)
                    qx, qy, qz, qw = (trans.transform.rotation.x, 
                                      trans.transform.rotation.y, 
                                      trans.transform.rotation.z, 
                                      trans.transform.rotation.w)
                    
                    print(f'{x:.2f};{y:.2f};{z:.2f}')
                    print(f'{qx:.2f};{qy:.2f};{qz:.2f};{qw:.2f}')
                except Exception as e:
                    print(e)

                # This will adjust as needed per iteration
                # loop_rate.sleep()

        except KeyboardInterrupt:
            pass


def main():
    node = StateListener()


if __name__ == '__main__':
    main()

