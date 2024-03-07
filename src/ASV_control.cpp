#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "ros_gz_interfaces/msg/param_vec.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/msg/quaternion.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Scalar.h>
#include <cmath>
#include <vector>


using namespace std;

class USVMovementNode : public rclcpp::Node {
public:
    USVMovementNode() : Node("usv_simple_movement") {
        // Constant Variables
        velocity_ASV_set = 0.0;
        omega_ASV_set = 0.0;
        Kp_velo = 500;
        Kp_omega = 500;
        Ki_velo = 0;
        Kd_velo = 100;
        K_Bearing = 15;
        k_alpha = 10;
        k_beta = -10;
        last_time_ = 0.0;
        last_x_ = 0.0;
        last_y_ = 0.0;
        last_yaw_ = 0.0;  
        prev_error_velocity = 0.0;
        prev_error_omega = 0.0;
        integral_velo = 0.0;
        integral_omega = 0.0;
        // Initialize goals vector with values
        goals.push_back({-100, 400});
        goals.push_back({350, -400});
        goals.push_back({800, 0});
        goal_number = 0;
        latest_bearing_ = 0.0;
        latest_range_ = 0.0;

        // Initialize publishers
        w1_left_pub_1 = create_publisher<std_msgs::msg::Float64>("/wamv_m_5/thrusters/left/thrust", 1);
        w2_right_pub_1 = create_publisher<std_msgs::msg::Float64>("/wamv_m_5/thrusters/right/thrust", 1);
        angle_r1_pub_1 = create_publisher<std_msgs::msg::Float64>("/wamv_m_5/thrusters/left/pos", 1);
        angle_r2_pub_1 = create_publisher<std_msgs::msg::Float64>("/wamv_m_5/thrusters/right/pos", 1);

        // Initialize subscribers
        range_boat_subscriber = this->create_subscription<ros_gz_interfaces::msg::ParamVec>(
            "/wamv_m_5/sensors/acoustics/receiver/range_bearing", 1,
            std::bind(&USVMovementNode::paramVecCallback, this, std::placeholders::_1));

        // Initialize subscribers
        pose_boat_subscriber = this->create_subscription<nav_msgs::msg::Odometry>(
            "/wamv_m_5/sensors/position/ground_truth_odometry", 1,
            std::bind(&USVMovementNode::poseBoatCallback, this, std::placeholders::_1));
    }

private:
    // Member variables to store constants
    double velocity_ASV_set;
    double omega_ASV_set;
    double Kp_velo;
    double Kp_omega;
    double Ki_velo;
    double Kd_velo;
    double K_Bearing;
    double k_alpha;
    double k_beta;
    double last_time_;
    double last_x_;
    double last_y_;
    double last_yaw_;
    double prev_error_velocity;
    double prev_error_omega;
    double integral_velo;
    double integral_omega;
    double latest_bearing_;
    double latest_range_;
    std::vector<std::vector<double>> goals;
    int goal_number;
    
    

    // Publishers
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr w1_left_pub_1;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr w2_right_pub_1;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr angle_r1_pub_1;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr angle_r2_pub_1;

    // Subscriber
    rclcpp::Subscription<ros_gz_interfaces::msg::ParamVec>::SharedPtr range_boat_subscriber;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_boat_subscriber;

    // // Timer for control loop
    // rclcpp::TimerBase::SharedPtr timer_;

    void paramVecCallback(const ros_gz_interfaces::msg::ParamVec::SharedPtr msg)
    {
        // Process the received message
        double range_value = msg->params[2].value.double_value; 
        double bearing_value = msg->params[1].value.double_value;   
        double sim_time_s = msg->header.stamp.sec + (msg->header.stamp.nanosec)/10^9;
        
        // //save beaning and range to variables for controller to use for logic calcs.
        // latest_range_ = range_value;
        // latest_bearing_ = bearing_value;

    }


    void poseBoatCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
            // Process the received message
        double current_time = msg->header.stamp.sec + (msg->header.stamp.nanosec) / std::pow(10.0, 9);
        
        //Subcribe to pose info.
        double current_x = msg->pose.pose.position.x;
        double current_y = msg->pose.pose.position.y;

        //Subcribe to Quaternion data
        geometry_msgs::msg::Quaternion quaternion_msg = msg->pose.pose.orientation;

        // Transform the quaternion to YPR
        tf2::Quaternion tf_quaternion;
        tf2::fromMsg(quaternion_msg, tf_quaternion);
        tf2Scalar yaw, pitch, roll;
        tf2::Matrix3x3(tf_quaternion).getEulerYPR(yaw, pitch, roll);

        double current_yaw = static_cast<double>(yaw);

        cout<<current_yaw<<endl;

        
        //  // Calculate velocity using finite differencing
        // double dt = current_time - last_time_;
        // double dx = current_x - last_x_;
        // double dy = current_y - last_y_;
        // double d_omega = current_yaw - last_yaw_;

        // double velocity_x = dx / dt;
        // double velocity_y = dy / dt;

        // double velocity = sqrt(pow(velocity_x, 2) + pow(velocity_y, 2));
        // double velocity_yaw = d_omega / dt;

        // cout<<velocity_yaw<<endl;

        // // Update last values for the next iteration
        // last_time_ = current_time;
        // last_x_ = current_x;
        // last_y_ = current_y;
        // last_yaw_ = current_yaw;

        //I can get velocities from the topic. They publish a twist. I 
        //I will keep the code uptop because it is a good ref for me. 

        double velo_x = msg->twist.twist.linear.x;
        double velo_y = msg->twist.twist.linear.y;
        double omega = msg->twist.twist.angular.z;

        //Send Pose to Velocity Controller.
        setpointValues(current_x, current_y, current_yaw);

        
        // Do something with the calculated velocities
        thursterController(velo_x, velo_y, omega);        
       
    }

    double circle_minus(double angle_1) {
        // Calculates the number of rotations past the E[-pi, pi].
        double num = std::floor(std::abs(angle_1 / (2 * M_PI)));

        // Check to see difference is the angles is positive or negative.
        double angle;
        if (angle_1 > 0) {
            angle = angle_1 - (num * 2 * M_PI);
        } else {
            angle = angle_1 + (num * 2 * M_PI);
        }

        // Checks the angle to ensure the smallest rotation is presented.

        if (angle > M_PI) {
            angle = (angle - 2 * M_PI);
        } else if (angle < -M_PI) {
            angle = (angle + 2 * M_PI);
        }
        return angle;
    }

    void setpointValues(double x, double y, double bearing) {
        // Calculate distances


        double d_x_boat = goals[goal_number][0] - x;
        double d_y_boat = goals[goal_number][1] - y;
        double rho = sqrt(pow(d_y_boat, 2) + pow(d_x_boat, 2));

        cout<<"range to goal is " << rho << endl;

        // Control Law to Calc Velocity
        double velocity_hold = K_Bearing * rho;

        if (velocity_hold > 6){
            velocity_hold = 6;
        }else if (velocity_hold < 2)
        {
            velocity_hold = 2;
        }
        
        double alpha_hold = -bearing + atan2(d_y_boat, d_x_boat);
        alpha_hold = circle_minus(alpha_hold);

        // double beta_hold = -bearing - alpha_hold;
        double beta_hold = 0;
        beta_hold = circle_minus(beta_hold);

        double omega_hold = k_alpha * alpha_hold + k_beta * beta_hold;

        cout<<"Bearning Error: "<< alpha_hold << endl;
        cout<<"atan2 Error: "<< atan2(d_y_boat, d_x_boat) << endl;


        // if (omega_hold > 1){
        //     omega_hold = 1;
        // }else if (omega_hold < -1)
        // {
        //     omega_hold < -1;
        // }
        
        // Assign the calculated values
        velocity_ASV_set = velocity_hold;
        omega_ASV_set = omega_hold;
        latest_range_ = rho;
    }


    // void setpointValues(double x, double y, double bearing) {
    //     // Allocate memory for ptr on the heap
    //     double* d_x_boat = new double(x_goal - x);

    //     double* d_y_boat = new double(y_goal - y);

    //     double* rho = new double(sqrt(pow(*d_y_boat, 2) + pow(*d_x_boat, 2)));

    //     // Control Law to Calc Velocity
    //     double* velocity_hold = new double(K_Bearing * (*rho));
    //     double* alpha_hold = new double(-bearing + atan2(*d_y_boat, *d_x_boat));
    //     double* beta_hold = new double(-bearing - *alpha_hold);
    //     double* omega_hold = new double(k_alpha*(*alpha_hold) + k_beta**beta_hold);

    //     // Assign the calculated value to velocity_ASV_set
    //     velocity_ASV_set = *velocity_hold;
    //     omega_ASV_set = *omega_hold;

    //     // Deallocate memory for ptr and ptr_2
    //     delete d_x_boat;
    //     delete d_y_boat;
    //     delete rho;
    //     delete velocity_hold;
    //     delete alpha_hold;
    //     delete beta_hold;
    //     delete omega_hold;
    // }



    void thursterController(double velocity_x, double velocity_y, double omega_)
    {
        if (latest_range_ < 20) {
            velocity_ASV_set = 0;
            goal_number += 1;
        }

        

        double error_velocity_x =  velocity_ASV_set - velocity_x;
        integral_velo += error_velocity_x;
        double derivative_velo = error_velocity_x - prev_error_velocity;

        cout<<"Velocity Error"<< error_velocity_x << endl;

 
        double thurst_velo = error_velocity_x*Kp_velo + Ki_velo * integral_velo + Kd_velo * derivative_velo;

        double error_omega_ =  omega_ASV_set - omega_;
        integral_omega += error_omega_;
        double derivative_omega = error_omega_ - prev_error_omega;

        cout<<"omega Error"<< error_omega_ << endl;

 
        double turing_velo_ = error_omega_*Kp_omega + Kd_velo * derivative_omega;
        // double turing_velo_;

        
        // if (thurst_velo < 100){
        //     double turing_velo_ = 0;
        // } else{
        //     double turing_velo_ = omega_ASV_set*K_Bearing;
        // }
        

        double thurst_velo_left_ = thurst_velo - turing_velo_;
        double thurst_velo_right_ = thurst_velo + turing_velo_;

        std_msgs::msg::Float64 thurst_velo_msg_left_;
        std_msgs::msg::Float64 thurst_velo_msg_right_;

        thurst_velo_msg_left_.data = thurst_velo_left_;
        thurst_velo_msg_right_.data = thurst_velo_right_;
        // std_msgs::msg::Float32 angle_r1_msg;
        // std_msgs::msg::Float32 angle_r2_msg;

        // Populate messages with data...

        w1_left_pub_1->publish(thurst_velo_msg_left_);
        w2_right_pub_1->publish(thurst_velo_msg_right_);


        // angle_r1_pub->publish(angle_r1_msg);
        // angle_r2_pub->publish(angle_r2_msg);

        prev_error_velocity = error_velocity_x;
        prev_error_omega = error_omega_;
    }


//==================================================================================================
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<USVMovementNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
