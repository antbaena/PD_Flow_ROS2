#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <Eigen/Core>
#include <cv_bridge/cv_bridge.h>
#include "scene_flow_visualization.h" // Incluye el archivo .h proporcionado
#include <pd_flow_msgs/msg/combined_image.hpp>
#include <pd_flow_msgs/msg/flow_field.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

using namespace std;

const unsigned int rows = 480; // Ajustar según tu imagen
const unsigned int cols = 640; // Ajustar según tu imagen
bool initialized = false;
int cont = 0;

class PDFlowNode : public rclcpp::Node
{
public:
    PDFlowNode() : Node("PD_flow_node"), pd_flow_(1, 30, 480)
    {
        cout << "Initializing PD_flow" << endl;

        // pd_flow_.initializePDFlow();
        subscription_ = this->create_subscription<pd_flow_msgs::msg::CombinedImage>(
            "/combined_image", 10, std::bind(&PDFlowNode::topic_callback, this, std::placeholders::_1));

        point_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("point_cloud", 10);
        vector_field_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("vector_field", 10);

        flow_pub_ = this->create_publisher<pd_flow_msgs::msg::FlowField>("flow_field", 10);
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
        if (cont == 0)
        {
            RCLCPP_INFO(this->get_logger(), "Iniciando el flujo flujo óptico cogiendo 2 imagenes iniciales...");
            pd_flow_.initializePDFlow();
            pd_flow_.process_frame(rgb_image_, depth_image_);
            pd_flow_.createImagePyramidGPU();
        }
        // else if (cont == 1) // Este if no es necesario
        //{
        //     pd_flow_.process_frame(rgb_image_, depth_image_);
        //     pd_flow_.createImagePyramidGPU();
        //     pd_flow_.solveSceneFlowGPU();
        // }
        else
        {
            // RCLCPP_INFO(this->get_logger(), "Calculando flujo óptico...");
            // Pasar las imágenes a PD_flow
            pd_flow_.process_frame(rgb_image_, depth_image_);
            pd_flow_.createImagePyramidGPU();
            pd_flow_.solveSceneFlowGPU();
            // RCLCPP_INFO(this->get_logger(), "Calculando flujo óptico...");

            // Publicar los resultados
            pd_flow_.updateScene();
            // publish_point_cloud();
            publish_motion_field();
        }
        cont++;
        // Reiniciar las imágenes después de procesarlas
        rgb_image_ = cv::Mat();
        depth_image_ = cv::Mat();
    }

    void publish_point_cloud()
    {
        std::vector<cv::Point3f> points;
        std::vector<cv::Point3f> vectors;

        pd_flow_.processPointCloud(points, vectors);

        sensor_msgs::msg::PointCloud2 point_cloud_msg;
        point_cloud_msg.header.frame_id = "map";
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

    void publish_flow_field()
    {
        std::vector<cv::Point3f> points;
        std::vector<cv::Point3f> vectors;

        pd_flow_.processPointCloud(points, vectors);

        auto msg = pd_flow_msgs::msg::FlowField();
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = "camera_frame";

        // Necesitamos desreferenciar el puntero para obtener el objeto subyacente
        if (initial_combined_image)
        {
            msg.image = *initial_combined_image;
        }

        // Aplanar las matrices de movimiento y copiar los datos
        size_t num_elements = pd_flow_.dx[0].size();
        msg.dx.resize(num_elements);
        msg.dy.resize(num_elements);
        msg.dz.resize(num_elements);

        // Variables para la suma total de dx, dy, dz
        float sum_dx = 0.0f;
        float sum_dy = 0.0f;
        float sum_dz = 0.0f;

        for (size_t i = 0; i < num_elements; ++i)
        {
            msg.dx[i] = pd_flow_.dx[0](i);
            msg.dy[i] = pd_flow_.dy[0](i);
            msg.dz[i] = pd_flow_.dz[0](i);

            // Acumulando la suma de dx, dy, dz
            sum_dx += msg.dx[i];
            sum_dy += msg.dy[i];
            sum_dz += msg.dz[i];
        }

        // Determinar si todos los vectores son nulos
        bool all_zero = (sum_dx == 0.0f && sum_dy == 0.0f && sum_dz == 0.0f);

        std::cout << "Publicando flujo óptico datos: " << msg.dx.size() << " elementos"
                  << (all_zero ? " - Todos los vectores son 0" : " - Vectores no nulos") << std::endl;

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
        for (unsigned int v = 0; v < rows; ++v)
        {
            for (unsigned int u = 0; u < cols; ++u)
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
        for (unsigned int v = 0; v < rows; ++v)
        {
            for (unsigned int u = 0; u < cols; ++u)
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
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vector_field_publisher_;

    PD_flow pd_flow_;
    cv::Mat rgb_image_;
    cv::Mat depth_image_;

    pd_flow_msgs::msg::CombinedImage::SharedPtr initial_combined_image;
    pd_flow_msgs::msg::CombinedImage::SharedPtr final_combined_image;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PDFlowNode>();

    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}
