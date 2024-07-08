#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <Eigen/Core>
#include <cv_bridge/cv_bridge.h>
#include "scene_flow_visualization.h" // Incluye el archivo .h proporcionado
#include <pd_flow_msgs/msg/combined_image.hpp>
#include <pd_flow_msgs/msg/flow_field.hpp>

using namespace std;
using Eigen::MatrixXf;
CSF_cuda csf_host, *csf_device;
MatrixXf rgb_matrix;
MatrixXf depth_matrix;
unsigned int cam_mode;
unsigned int rows;
unsigned int cols;
unsigned int ctf_levels;
unsigned int pyr_levels;

class PDFlowNode : public rclcpp::Node
{
public:
    PDFlowNode() : Node("PD_flow_node"), pd_flow_(1, 30, 240)
    {
        subscription_ = this->create_subscription<pd_flow_msgs::msg::CombinedImage>(
            "combined_image", 10, std::bind(&PDFlowNode::topic_callback, this, std::placeholders::_1));
        flow_pub_ = this->create_publisher<pd_flow_msgs::msg::FlowField>("flow_field", 10);
    }

private:
    void topic_callback(const pd_flow_msgs::msg::CombinedImage::SharedPtr msg)
    {
        // Convertir imágenes ROS a OpenCV
        cv_bridge::CvImagePtr rgb_cv_ptr;
        cv_bridge::CvImagePtr depth_cv_ptr;

        try
        {
            rgb_cv_ptr = cv_bridge::toCvCopy(msg->rgb_image, sensor_msgs::image_encodings::BGR8);
            depth_cv_ptr = cv_bridge::toCvCopy(msg->depth_image, sensor_msgs::image_encodings::TYPE_32FC1);
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        rgb_image_ = rgb_cv_ptr->image;
        depth_image_ = depth_cv_ptr->image;
        process_images();
    }

    void process_images()
    {
        if (!rgb_image_.empty() && !depth_image_.empty())
        {
            // Pasar las imágenes a PD_flow
            if (pd_flow_.GetFromRGBDImages(rgb_image_, depth_image_))
            {
                pd_flow_.createImagePyramidGPU();
                pd_flow_.solveSceneFlowGPU();

                // Publicar los resultados
                publish_flow_field();
            }
        }
    }

    void publish_flow_field()
    {
        auto msg = pd_flow_msgs::msg::FlowField();
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = "camera_frame";

        // Aplanar las matrices de movimiento y copiar los datos
        size_t num_elements = pd_flow_.dx[0].size();
        msg.dx.resize(num_elements);
        msg.dy.resize(num_elements);
        msg.dz.resize(num_elements);

        for (size_t i = 0; i < num_elements; ++i)
        {
            msg.dx[i] = pd_flow_.dx[0](i);
            msg.dy[i] = pd_flow_.dy[0](i);
            msg.dz[i] = pd_flow_.dz[0](i);
        }

        flow_pub_->publish(msg);
    }

    rclcpp::Subscription<pd_flow_msgs::msg::CombinedImage>::SharedPtr subscription_;
    rclcpp::Publisher<pd_flow_msgs::msg::FlowField>::SharedPtr flow_pub_;
    cv::Mat rgb_image_, depth_image_;
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
