#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <memory>
#include "pdflow_cudalib.h"
#include <Eigen/Core>
#include <pd_flow_msgs/msg/combined_image.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace std;
const unsigned int rows = 480; // Ajustar según tu imagen
const unsigned int cols = 640; // Ajustar según tu imagen
class PDFlowNode : public rclcpp::Node
{

public:
    PD_flow() : Node("PD_flow")
    {
        cout << "Initializing PD_flow" << endl;
        subscription_ = this->create_subscription<pd_flow_msgs::msg::CombinedImage>(
            "/combined_image", 10, std::bind(&PDFlowNode::topic_callback, this, std::placeholders::_1));

        flow_pub_ = this->create_publisher<pd_flow_msgs::msg::FlowField>("flow_field", 10);
    }

private:
    void topic_callback(const pd_flow_msgs::msg::CombinedImage msg)
    {
        cout << "Recibiendo imágenes..." << endl;
        // Convertir imágenes ROS a OpenCV
        cv_bridge::CvImagePtr rgb_cv_ptr;
        cv_bridge::CvImagePtr depth_cv_ptr;
        try
        {
            // Convertir el mensaje de ROS a imágenes OpenCV
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

    void process_images()
    {
        RCLCPP_INFO(this->get_logger(), "Calculando flujo óptico...");

        // Pasar las imágenes a PD_flow
        if (pd_flow_.GetFromRGBDImages(rgb_image_, depth_image_))
        {
            RCLCPP_INFO(this->get_logger(), "Comenza a calcular flujo óptico...");
            pd_flow_.createImagePyramidGPU();
            RCLCPP_INFO(this->get_logger(), "Piramide calculada con exito");
            pd_flow_.solveSceneFlowGPU();
            RCLCPP_INFO(this->get_logger(), "Flujo óptico calculado con exito");

            // Publicar los resultados
            publish_flow_field();
        }

        // Reiniciar las imágenes después de procesarlas
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

        std::cout << "Publicando flujo óptico datos: " << msg.dx.size() << " elementos" << std::endl;
        flow_pub_->publish(msg);

        // Crear imagen OpenCV del flujo óptico
        cv::Mat flow_image = createImage();

        // Mostrar imagen con OpenCV
        cv::imshow("Optical Flow", flow_image);
        cv::waitKey(1); // Esperar un milisegundo para que se actualice la ventana
    }

    cv::Mat createImage() const
    {
        // Crear imagen RGB (una color por dirección)
        cv::Mat sf_image(rows, cols, CV_8UC3);

        // Calcular los valores máximos del flujo (de sus componentes)
        float maxmodx = 0.f, maxmody = 0.f, maxmodz = 0.f;
        for (unsigned int v = 0; v < rows; v++)
        {
            for (unsigned int u = 0; u < cols; u++)
            {
                size_t index = v + u * rows;
                if (fabs(pd_flow_.dx[0](index)) > maxmodx)
                    maxmodx = fabs(pd_flow_.dx[0](index));
                if (fabs(pd_flow_.dy[0](index)) > maxmody)
                    maxmody = fabs(pd_flow_.dy[0](index));
                if (fabs(pd_flow_.dz[0](index)) > maxmodz)
                    maxmodz = fabs(pd_flow_.dz[0](index));
            }
        }

        // Crear una representación RGB del flujo óptico
        for (unsigned int v = 0; v < rows; v++)
        {
            for (unsigned int u = 0; u < cols; u++)
            {
                size_t index = v + u * rows;
                sf_image.at<cv::Vec3b>(v, u)[0] = static_cast<unsigned char>(255.f * fabs(pd_flow_.dx[0](index)) / maxmodx); // Azul - x
                sf_image.at<cv::Vec3b>(v, u)[1] = static_cast<unsigned char>(255.f * fabs(pd_flow_.dy[0](index)) / maxmody); // Verde - y
                sf_image.at<cv::Vec3b>(v, u)[2] = static_cast<unsigned char>(255.f * fabs(pd_flow_.dz[0](index)) / maxmodz); // Rojo - z
            }
        }

        return sf_image;
    }

    rclcpp::Subscription<pd_flow_msgs::msg::CombinedImage>::SharedPtr subscription_;
    rclcpp::Publisher<pd_flow_msgs::msg::FlowField>::SharedPtr flow_pub_;
    PD_flow pd_flow_;
    cv::Mat rgb_image_;
    cv::Mat depth_image_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PD_flow>();

    while (rclcpp::ok())
    {
        rclcpp::spin(node);
        // imagenRGB, imnagenDepth = sceneflow.CaptureFrame();
        // sceneflow.createImagePyramidGPU(rgb, depth);
        // sceneflow.solveSceneFlowGPU();
    }
    rclcpp::shutdown();
    return 0;
}

// void createImagePyramidGPU(MatrixXf colour_wf, MatrixXf depth_wf)
// {
//     // Copy new frames to the scene flow object
//     csf_host.copyNewFrames(colour_wf.data(), depth_wf.data());

//     // Copy scene flow object to device
//     csf_device = ObjectToDevice(&csf_host);
//     // Cam mode es sobre la resolucion de la camara
//     unsigned int cam_mode = 1;
//     unsigned int rows = colour_wf.rows();
//     unsigned int cols = colour_wf.cols();
//     unsigned int ctf_levels = round(log2(rows / 15)) + 1;
//     unsigned int pyr_levels = round(log2(640 / (cam_mode * cols))) + ctf_levels;

//     GaussianPyramidBridge(csf_device, pyr_levels, cam_mode);

//     // Copy scene flow object back to host
//     BridgeBack(&csf_host, csf_device);
// }

// void solveSceneFlowGPU()
// {
//     // Define variables

//     unsigned int s;
//     unsigned int cols_i, rows_i;
//     unsigned int level_image;
//     unsigned int num_iter;

//     // For every level (coarse-to-fine)
//     for (unsigned int i = 0; i < ctf_levels; i++)
//     {
//         const unsigned int width = colour_wf.cols();
//         s = pow(2.f, int(ctf_levels - (i + 1)));
//         cols_i = cols / s;
//         rows_i = rows / s;
//         level_image = ctf_levels - i + round(log2(width / cols)) - 1;

//         //=========================================================================
//         //                              Cuda - Begin
//         //=========================================================================

//         // Cuda allocate memory
//         csf_host.allocateMemoryNewLevel(rows_i, cols_i, i, level_image);

//         // Cuda copy object to device
//         csf_device = ObjectToDevice(&csf_host);

//         // Assign zeros to the corresponding variables
//         AssignZerosBridge(csf_device);

//         // Upsample previous solution
//         if (i > 0)
//             UpsampleBridge(csf_device);

//         // Compute connectivity (Rij)
//         RijBridge(csf_device);

//         // Compute colour and depth derivatives
//         ImageGradientsBridge(csf_device);
//         WarpingBridge(csf_device);

//         // Compute mu_uv and step sizes for the primal-dual algorithm
//         MuAndStepSizesBridge(csf_device);

//         // Primal-Dual solver
//         for (num_iter = 0; num_iter < num_max_iter[i]; num_iter++)
//         {
//             GradientBridge(csf_device);
//             DualVariablesBridge(csf_device);
//             DivergenceBridge(csf_device);
//             PrimalVariablesBridge(csf_device);
//         }

//         // Filter solution
//         FilterBridge(csf_device);

//         // Compute the motion field
//         MotionFieldBridge(csf_device);

//         // BridgeBack
//         BridgeBack(&csf_host, csf_device);

//         // Free variables of variables associated to this level
//         csf_host.freeLevelVariables();

//         // Copy motion field and images to CPU
//         csf_host.copyAllSolutions(dx[ctf_levels - i - 1].data(), dy[ctf_levels - i - 1].data(), dz[ctf_levels - i - 1].data(),
//                                   depth[level_image].data(), depth_old[level_image].data(), colour[level_image].data(), colour_old[level_image].data(),
//                                   xx[level_image].data(), xx_old[level_image].data(), yy[level_image].data(), yy_old[level_image].data());

//         //=========================================================================
//         //                              Cuda - end
//         //=========================================================================
//     }
// }
