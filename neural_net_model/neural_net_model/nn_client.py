from os import write
import sys
from tg_interfaces.srv import ReverseKinematic
import rclpy
from rclpy.node import Node
from csv import writer

class NN_client(Node):
    """
    Cliente com propósito de requisitar juntas do 
    modelo de rede neural dados posição e orientação finais
    """
    def __init__(self):
        super().__init__('nn_client')
        self.cli = self.create_client(ReverseKinematic, 'reverse_kinematic')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        self.req = ReverseKinematic.Request()

    def send_request(self):
        self.req.x = float(sys.argv[1])
        self.req.y = float(sys.argv[2])
        self.req.z = float(sys.argv[3])
        self.req.qx = float(sys.argv[4])
        self.req.qy = float(sys.argv[5])
        self.req.qz = float(sys.argv[6])
        self.req.qw = float(sys.argv[7])

        self.future = self.cli.call_async(self.req)

def main():
    rclpy.init()
    client = NN_client()
    client.send_request()

    while rclpy.ok():
        rclpy.spin_once(client)
        if client.future.done():
            try:
                response = client.future.result()
            except Exception as e:
                    client.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                j1 = response.joint1
                j2 = response.joint2
                j3 = response.joint3
                j4 = response.joint4
                j5 = response.joint5
                client.get_logger().info(
                    f'Joints returned by model: {j1};{j2};{j3};{j4};{j5}')
                
                    
            break

    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
