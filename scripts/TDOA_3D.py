#!/usr/bin/env python3

# import rclpy
# from vrx_ros.module_to_import import TrilaterationSolver

#Author: Michael MacGillivray
#This code will use multilateration to solve to the pose of a target ASV. 
#ROS2 Humble and Gazebo Garden are used. 
#The original Ros Packaget is the VRX_ws from git. 


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import math
from geometry_msgs.msg import Quaternion, Point
from tf2_ros import TransformListener
import tf2_py
import matplotlib.pyplot as plt

import numpy as np
import scipy.optimize as opt
from scipy.optimize import least_squares
import csv

class TrilaterationSolver(Node):
    def __init__(self):
        super().__init__('trilateration_solver')

        # Initialize variables to store the latest positions for boats and the target
        self.boat_1_positions = 0  # List to store x, y, z values for each boat
        self.boat_2_positions = 0  # List to store x, y, z values for each boat
        self.boat_3_positions = 0  # List to store x, y, z values for each boat
        self.boat_4_positions = 0  # List to store x, y, z values for each boat


        self.target_position = 0  # Tuple to store x, y, z values for the target
        self.actual_x = []
        self.actual_y = []
        self.estimated_x = []
        self.estimated_y = []

        self.x_last = 0
        self.y_last = 0
        self.z_last = -0.10

        # Create subscribers for all boats
        self.boat_1_subscriber = self.create_subscription(
            Odometry,
            '/wamv/sensors/position/ground_truth_odometry',
            self.pose_boat_1_callback,
            1
        )
        
        # Create subscribers for all boats
        self.boat_2_subscriber = self.create_subscription(
            Odometry,
            '/wamv_m_2/sensors/position/ground_truth_odometry',
            self.pose_boat_2_callback,
            1
        )
        
        # Create subscribers for all boats
        self.boat_3_subscriber = self.create_subscription(
            Odometry,
            '/wamv_m_3/sensors/position/ground_truth_odometry',
            self.pose_boat_3_callback,
            1
        )
        
        # Create subscribers for all boats
        self.boat_4_subscriber = self.create_subscription(
            Odometry,
            '/wamv_m_4/sensors/position/ground_truth_odometry',
            self.pose_boat_4_callback,
            1
        )


        # Subscribe to target position
        self.pose_target_subscriber = self.create_subscription(
            Odometry,
            '/wamv_m_5/sensors/position/ground_truth_odometry',
            self.pose_target_callback,
            1
        )

        # Create a publisher for estimated location
        self.estimated_location_publisher = self.create_publisher(
            Point,
            '/target/estimated_location',
            10
        )


        self.signal_send = self.create_timer(3, self.solve)

        # self.noise_std_dev = 1

    def pose_boat_1_callback(self, msg):        
        # Process the received message
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        current_z = msg.pose.pose.position.z

        self.boat_1_positions = np.array([current_x, current_y, current_z])
    
    def pose_boat_2_callback(self, msg):        
        # Process the received message
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        current_z = msg.pose.pose.position.z

        self.boat_2_positions = np.array([current_x, current_y, current_z])
    
    def pose_boat_3_callback(self, msg):        
        # Process the received message
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        current_z = msg.pose.pose.position.z

        self.boat_3_positions = np.array([current_x, current_y, current_z])
    
    def pose_boat_4_callback(self, msg):        
        # Process the received message
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        current_z = msg.pose.pose.position.z

        self.boat_4_positions = np.array([current_x, current_y, current_z])

    def pose_target_callback(self, msg):
        # Process the received message
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        current_z = msg.pose.pose.position.z
        # Update the latest x, y, z values for the target
        self.target_position = np.array([current_x, current_y, current_z])

    def publish_estimated_location(self, x, y, z):
        # Create a Point message
        point_msg = Point()
        point_msg.x = x
        point_msg.y = y
        point_msg.z = z

        # Publish the estimated location
        self.estimated_location_publisher.publish(point_msg)
       
    def functions(self, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, d01, d02, d03, d12, d13, d23):
        def fn(args):
            x, y, z = args
            a = np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2) - np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) - d01
            b = np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2) - np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) - d02
            c = np.sqrt((x - x3) ** 2 + (y - y3) ** 2 + (z - z3) ** 2) - np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) - d03
            d = np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2) - np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2) - d12
            e = np.sqrt((x - x3) ** 2 + (y - y3) ** 2 + (z - z3) ** 2) - np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2) - d13
            f = np.sqrt((x - x3) ** 2 + (y - y3) ** 2 + (z - z3) ** 2) - np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2) - d23
            return [a, b, c, d, e, f]
        return fn

    def jacobian(self, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3):
        def fn(args):
            x, y, z = args
            adx = (x - x1) / np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2) - (x - x0) / np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)
            bdx = (x - x2) / np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2) - (x - x0) / np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)
            cdx = (x - x3) / np.sqrt((x - x3) ** 2 + (y - y3) ** 2 + (z - z3) ** 2) - (x - x0) / np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2)
            ady = (y - y1) / np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2) - (y - y0) / np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)
            bdy = (y - y2) / np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2) - (y - y0) / np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)
            cdy = (y - y3) / np.sqrt((x - x3) ** 2 + (y - y3) ** 2 + (z - z3) ** 2) - (y - y0) / np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2)
            adz = (z - z1) / np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2) - (z - z0) / np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)
            bdz = (z - z2) / np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2) - (z - z0) / np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)
            cdz = (z - z3) / np.sqrt((x - x3) ** 2 + (y - y3) ** 2 + (z - z3) ** 2) - (z - z0) / np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2)

            ddx = (x - x2) / np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2) - (x - x0) / np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
            edx = (x - x3) / np.sqrt((x - x3) ** 2 + (y - y3) ** 2 + (z - z3) ** 2) - (x - x0) / np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
            fdx = (x - x3) / np.sqrt((x - x3) ** 2 + (y - y3) ** 2 + (z - z3) ** 2) - (x - x0) / np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
            ddy = (y - y2) / np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2) - (y - y0) / np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
            edy = (y - y3) / np.sqrt((x - x3) ** 2 + (y - y3) ** 2 + (z - z3) ** 2) - (y - y0) / np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
            fdy = (y - y3) / np.sqrt((x - x3) ** 2 + (y - y3) ** 2 + (z - z3) ** 2) - (y - y0) / np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
            ddz = (z - z2) / np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2) - (z - z0) / np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
            edz = (z - z3) / np.sqrt((x - x3) ** 2 + (y - y3) ** 2 + (z - z3) ** 2) - (z - z0) / np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
            fdz = (z - z3) / np.sqrt((x - x3) ** 2 + (y - y3) ** 2 + (z - z3) ** 2) - (z - z0) / np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)

            return np.array([
                [adx, ady, adz],
                [bdx, bdy, bdz],
                [cdx, cdy, cdz],
                [ddx, ddy, ddz],
                [edx, edy, edz],
                [fdx, fdy, fdz]])
        return fn


    def solve(self):
        # Extract x and y coordinates for each boat

        x0, y0, z0 = self.boat_1_positions[0], self.boat_1_positions[1], self.boat_1_positions[2]
        x1, y1, z1 = self.boat_2_positions[0], self.boat_2_positions[1], self.boat_2_positions[2]
        x2, y2, z2 = self.boat_3_positions[0], self.boat_3_positions[1], self.boat_3_positions[2]
        x3, y3, z3 = self.boat_4_positions[0], self.boat_4_positions[1], self.boat_4_positions[2]

        x_target, y_target, z_target = self.target_position[0], self.target_position[1], self.target_position[2]
  
        ranges = {}  # Dictionary to store the ranges for each boat

        range_0 = math.sqrt((x0 - x_target)**2 + (y0 - y_target)**2 + (z0 - z_target)**2)
        range_1 = math.sqrt((x1 - x_target)**2 + (y1 - y_target)**2 + (z1 - z_target)**2)
        range_2 = math.sqrt((x2 - x_target)**2 + (y2 - y_target)**2 + (z2 - z_target)**2)
        range_3 = math.sqrt((x3 - x_target)**2 + (y3 - y_target)**2 + (z3 - z_target)**2)

        d01 = range_1 - range_0 + np.random.normal(0, 10)
        d02 = range_2 - range_0 + np.random.normal(0, 10)
        d03 = range_3 - range_0 + np.random.normal(0, 10)
        d12 = range_2 - range_1 + np.random.normal(0, 10)
        d13 = range_3 - range_1 + np.random.normal(0, 10)
        d23 = range_3 - range_2 + np.random.normal(0, 10)


        xp = self.x_last
        yp = self.y_last
        zp = self.z_last

        F = self.functions(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, d01, d02, d03, d12, d13, d23)
        #J = self.jacobian(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3)

        #Soultion bounds for solver to adhere by (x_min, y_min, z_min). 
        bound = ([-np.inf, -np.inf, -np.inf],[np.inf, np.inf, 0])

        estimated_values = least_squares(F, x0=[xp, yp, zp], bounds=bound).x

        if estimated_values is not None:
            x, y, z = estimated_values
            print(f"x error: {x - x_target}")
            print(f"y error: {y - y_target}")
            print(f"z error: {z - z_target} \n")
            self.publish_estimated_location(x, y, z)
        else:
            return None
        
        self.x_last = x
        self.y_last = y
        self.z_last = z


        # Update the lists with the current x and y coordinates
        self.actual_x.append(x_target)
        self.actual_y.append(y_target)
        self.estimated_x.append(x)
        self.estimated_y.append(y)

        # Save the actual and estimated poses to a CSV file
        with open('/home/michael-asv/vrx_ws/src/vrx/vrx_ros/scripts/poses.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Actual X', 'Actual Y', 'Estimated X', 'Estimated Y'])
            for i in range(len(self.actual_x)):
                writer.writerow([self.actual_x[i], self.actual_y[i], self.estimated_x[i], self.estimated_y[i]])


def main(args=None):
    rclpy.init(args=args)
    trilateration_solver = TrilaterationSolver()
    try:
        rclpy.spin(trilateration_solver)
    except KeyboardInterrupt:
        pass
    trilateration_solver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
