from os import write
import sys
from tg_interfaces.srv import JointExample
import rclpy
from rclpy.node import Node
from csv import writer

class JointClient(Node):
    def __init__(self):
        super().__init__('joint_client')
        self.cli = self.create_client(JointExample, 'joint_example')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        self.req = JointExample.Request()

    def send_request(self):
        self.req.joint1 = float(sys.argv[1])
        self.req.joint2 = float(sys.argv[2])
        self.req.joint3 = float(sys.argv[3])
        self.req.joint4 = float(sys.argv[4])
        self.req.joint5 = float(sys.argv[5])

        self.future = self.cli.call_async(self.req)

def main():
    rclpy.init()
    joint_client = JointClient()
    joint_client.send_request()

    while rclpy.ok():
        rclpy.spin_once(joint_client)
        if joint_client.future.done():
            try:
                response = joint_client.future.result()
            except Exception as e:
                    joint_client.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                j1 = joint_client.req.joint1#livre
                j2 = joint_client.req.joint2#-2 a +2
                j3 = joint_client.req.joint3#-2 a +2
                j4 = joint_client.req.joint4#livre
                j5 = joint_client.req.joint5#-2 a +2
                x = response.x
                y = response.y
                z = response.z
                qx = response.qx
                qy = response.qy
                qz = response.qz
                qw = response.qw
                joint_client.get_logger().info(
                    f'Result of joints {j1};{j2};{j3};{j4};{j5}\n{x};{y};{z};{qx};{qy};{qz};{qw}')
                with open('/home/osboxes/results/data.csv', 'a+', newline='') as f:
                    csv_writer = writer(f)
                    csv_writer.writerow([
                        j1, j2, j3, j4, j5,
                        x, y, z, qx, qy, qz, qw
                    ])
                    
            break

    joint_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
