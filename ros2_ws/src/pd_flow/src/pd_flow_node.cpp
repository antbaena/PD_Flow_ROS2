#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <Eigen/Core>
#include <cv_bridge/cv_bridge.h>
#include "scene_flow_visualization.h"
#include <pd_flow_msgs/msg/combined_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pd_flow_msgs/msg/flow_field.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <chrono>
using namespace std;

const unsigned int rows = 480; // Adjust according to your image
const unsigned int cols = 640; // Adjust according to your image

class PDFlowNode : public rclcpp::Node
{
public:
    PDFlowNode() : Node("PD_flow_node"), pd_flow_(1, 30, 480), initialized(false), total_time(0), execution_count(0)
    {

        RCLCPP_INFO(this->get_logger(), "Initializing PD_flow");
        timer_ = this->create_wall_timer(
            std::chrono::seconds(4),
            std::bind(&PDFlowNode::log_timer_callback, this));

        subscription_ = this->create_subscription<pd_flow_msgs::msg::CombinedImage>(
            "/combined_image", 10, std::bind(&PDFlowNode::topic_callback, this, std::placeholders::_1));

        point_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("point_cloudd", 10);
        vector_field_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("vector_field", 10);

        flow_pub_ = this->create_publisher<pd_flow_msgs::msg::FlowField>("flow_field", 1000);
    }

private:
    void topic_callback(const pd_flow_msgs::msg::CombinedImage::SharedPtr msg)
    {
        try
        {
            if (!final_combined_image)
            {
                final_combined_image = msg;
            }
            else
            {
                initial_combined_image = final_combined_image;
                final_combined_image = msg;
            }

            // Convert the ROS message to OpenCV images
            rgb_image_ = cv_bridge::toCvCopy(msg->rgb_image, sensor_msgs::image_encodings::RGB8)->image;
            depth_image_ = cv_bridge::toCvCopy(msg->depth_image, sensor_msgs::image_encodings::TYPE_16UC1)->image;
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        process_images();
    }

    void log_timer_callback()
    {
        if (execution_count > 0)
        {
            double average_time = total_time / execution_count;
            RCLCPP_INFO(this->get_logger(), "Average time to process the images: %f seconds", average_time);

            // Reset the counters
            total_time = 0;
            execution_count = 0;
        }
    }

    void process_images()
    {
        if (!initialized)
        {
            RCLCPP_INFO(this->get_logger(), "Starting optical flow by processing the first two images...");
            pd_flow_.initializePDFlow();
            pd_flow_.process_frame(rgb_image_, depth_image_);
            pd_flow_.createImagePyramidGPU();
            initialized = true;
        }
        else
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            // Pass the images to PD_flow
            pd_flow_.process_frame(rgb_image_, depth_image_);
            pd_flow_.createImagePyramidGPU();
            pd_flow_.solveSceneFlowGPU();

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = end_time - start_time;

            // Accumulate the time and increment the execution count
            total_time += elapsed_time.count();
            execution_count++;

            // Publish the results
            publish_flow_field();

            // DEBUG -- Only for visualization --
            //  pd_flow_.updateScene();
            //  publish_point_cloud();
            //  publish_motion_field();
        }
    }

    void publish_flow_field()
    {
        auto msg = pd_flow_msgs::msg::FlowField();
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = "camera_frame";

        // Get the initial image sent by the topic (original RGB and depth images)
        if (initial_combined_image)
        {
            msg.image = *initial_combined_image;
        }

        // Flatten the motion matrices and copy the data
        size_t num_elements = pd_flow_.dx[0].size();
        msg.dx.resize(num_elements);
        msg.dy.resize(num_elements);
        msg.dz.resize(num_elements);

        const unsigned int repr_level = round(log2(pd_flow_.colour_wf.cols() / cols));

        long i = 0;
        for (unsigned int v = 0; v < rows; v++)
        {
            for (unsigned int u = 0; u < cols; u++)
            {
                msg.dx[i] = pd_flow_.dx[repr_level](v, u);
                msg.dy[i] = pd_flow_.dy[repr_level](v, u);
                msg.dz[i] = pd_flow_.dz[repr_level](v, u);

                i++;
            }
        }

        flow_pub_->publish(msg);
    }

    // From here onwards, it is unrevised

    void publish_point_cloud()
    {
        std::vector<cv::Point3f> points;
        std::vector<cv::Point3f> vectors;

        pd_flow_.processPointCloud(points, vectors);

        sensor_msgs::msg::PointCloud2 point_cloud_msg;
        point_cloud_msg.header.frame_id = "field";
        point_cloud_msg.header.stamp = this->now();
        point_cloud_msg.height = 1;
        point_cloud_msg.width = points.size();
        point_cloud_msg.is_dense = false;
        point_cloud_msg.is_bigendian = false;

        sensor_msgs::PointCloud2Modifier modifier(point_cloud_msg);
        modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

        sensor_msgs::PointCloud2Iterator<float> iter_x(point_cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(point_cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(point_cloud_msg, "z");

        for (size_t i = 0; i < points.size(); ++i, ++iter_x, ++iter_y, ++iter_z)
        {
            *iter_x = points[i].x;
            *iter_y = points[i].y;
            *iter_z = points[i].z;
        }

        point_cloud_publisher_->publish(point_cloud_msg);
    }

    void publish_motion_field()
    {
        std::vector<cv::Point3f> points;
        std::vector<cv::Point3f> vectors;

        pd_flow_.processPointCloud(points, vectors);
        visualization_msgs::msg::MarkerArray marker_array;
        int id = 0;
        for (size_t i = 0; i < vectors.size(); ++i)
        {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = this->now();
            marker.ns = "vector_field";
            marker.id = id++;
            marker.type = visualization_msgs::msg::Marker::ARROW;
            marker.action = visualization_msgs::msg::Marker::ADD;

            marker.pose.position.x = vectors[i].x;
            marker.pose.position.y = vectors[i].y;
            marker.pose.position.z = vectors[i].z;

            marker_array.markers.push_back(marker);
        }

        vector_field_publisher_->publish(marker_array);
    }

    // Declaration of topic subscriptions
    rclcpp::Subscription<pd_flow_msgs::msg::CombinedImage>::SharedPtr subscription_;

    // Declaration of topic publishers
    rclcpp::Publisher<pd_flow_msgs::msg::FlowField>::SharedPtr flow_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vector_field_publisher_;

    // Declaration of custom messages
    pd_flow_msgs::msg::CombinedImage::SharedPtr initial_combined_image;
    pd_flow_msgs::msg::CombinedImage::SharedPtr final_combined_image;

    // Declaration of internal variables
    cv::Mat rgb_image_;
    cv::Mat depth_image_;
    rclcpp::TimerBase::SharedPtr timer_;
    bool initialized;
    double total_time;
    int execution_count;

    PD_flow pd_flow_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PDFlowNode>();

    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}
