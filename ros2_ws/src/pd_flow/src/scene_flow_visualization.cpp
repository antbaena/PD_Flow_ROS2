
/*****************************************************************************
**                Primal-Dual Scene Flow for RGB-D cameras                  **
**                ----------------------------------------                  **
**                                                                          **
**  Copyright(c) 2015, Mariano Jaimez Tarifa, University of Malaga          **
**  Copyright(c) 2015, Mohamed Souiai, Technical University of Munich       **
**  Copyright(c) 2015, MAPIR group, University of Malaga                    **
**  Copyright(c) 2015, Computer Vision group, Tech. University of Munich    **
**                                                                          **
**  This program is free software: you can redistribute it and/or modify    **
**  it under the terms of the GNU General Public License (version 3) as     **
**  published by the Free Software Foundation.                              **
**                                                                          **
**  This program is distributed in the hope that it will be useful, but     **
**  WITHOUT ANY WARRANTY; without even the implied warranty of              **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            **
**  GNU General Public License for more details.                            **
**                                                                          **
**  You should have received a copy of the GNU General Public License       **
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.   **
**                                                                          **
*****************************************************************************/

#include "scene_flow_visualization.h"

PD_flow::PD_flow(unsigned int cam_mode_config, unsigned int fps_config, unsigned int rows_config)
{

    rows = 480;                 // rows_config; // Maximum size of the coarse-to-fine scheme - Default 240 (QVGA)
    cols = 640;                 // rows * 320 / 240;
    cam_mode = cam_mode_config; // (1 - 640 x 480, 2 - 320 x 240), Default - 1
    ctf_levels = round(log2(rows / 15)) + 1;
    fovh = M_PI * 62.5f / 180.f;
    fovv = M_PI * 45.f / 180.f;
    fps = fps_config; // In Hz, Default - 30

    // Iterations of the primal-dual solver at each pyramid level.
    // Maximum value set to 100 at the finest level
    for (int i = 5; i >= 0; i--)
    {
        if (i >= ctf_levels - 1)
            num_max_iter[i] = 100;
        else
            num_max_iter[i] = num_max_iter[i + 1] - 15;
    }

    // num_max_iter[ctf_levels-1] = 0.f;

    // Compute gaussian mask
    float v_mask[5] = {1.f, 4.f, 6.f, 4.f, 1.f};
    for (unsigned int i = 0; i < 5; i++)
        for (unsigned int j = 0; j < 5; j++)
            g_mask[i + 5 * j] = v_mask[i] * v_mask[j] / 256.f;

    // Matrices that store the original and filtered images with the image resolution
    colour_wf.resize(480 / cam_mode, 640 / cam_mode);
    depth_wf.resize(480 / cam_mode, 640 / cam_mode);

    // Resize vectors according to levels
    dx.resize(ctf_levels);
    dy.resize(ctf_levels);
    dz.resize(ctf_levels);

    const unsigned int width = colour_wf.cols();
    const unsigned int height = colour_wf.rows();
    unsigned int s, cols_i, rows_i;

    for (unsigned int i = 0; i < ctf_levels; i++)
    {
        s = pow(2.f, int(ctf_levels - (i + 1)));
        cols_i = cols / s;
        rows_i = rows / s;
        dx[ctf_levels - i - 1].resize(rows_i, cols_i);
        dy[ctf_levels - i - 1].resize(rows_i, cols_i);
        dz[ctf_levels - i - 1].resize(rows_i, cols_i);
    }

    // Resize pyramid
    const unsigned int pyr_levels = round(log2(width / cols)) + ctf_levels;
    colour.resize(pyr_levels);
    colour_old.resize(pyr_levels);
    depth.resize(pyr_levels);
    depth_old.resize(pyr_levels);
    xx.resize(pyr_levels);
    xx_old.resize(pyr_levels);
    yy.resize(pyr_levels);
    yy_old.resize(pyr_levels);

    for (unsigned int i = 0; i < pyr_levels; i++)
    {
        s = pow(2.f, int(i));
        colour[i].resize(height / s, width / s);
        colour_old[i].resize(height / s, width / s);
        colour[i].setZero();
        colour_old[i].setZero();
        depth[i].resize(height / s, width / s);
        depth_old[i].resize(height / s, width / s);
        depth[i].setZero();
        depth_old[i].setZero();
        xx[i].resize(height / s, width / s);
        xx_old[i].resize(height / s, width / s);
        xx[i].setZero();
        xx_old[i].setZero();
        yy[i].resize(height / s, width / s);
        yy_old[i].resize(height / s, width / s);
        yy[i].setZero();
        yy_old[i].setZero();
    }

    // Parameters of the variational method
    lambda_i = 0.04f;
    lambda_d = 0.35f;
    mu = 75.f;
}

void PD_flow::createImagePyramidGPU()
{
    // Copy new frames to the scene flow object
    csf_host.copyNewFrames(colour_wf.data(), depth_wf.data());

    // Copy scene flow object to device
    csf_device = ObjectToDevice(&csf_host);

    unsigned int pyr_levels = round(log2(640 / (cam_mode * cols))) + ctf_levels;
    GaussianPyramidBridge(csf_device, pyr_levels, cam_mode);

    // Copy scene flow object back to host
    BridgeBack(&csf_host, csf_device);
}

void PD_flow::solveSceneFlowGPU()
{
    // Define variables

    unsigned int s;
    unsigned int cols_i, rows_i;
    unsigned int level_image;
    unsigned int num_iter;

    // For every level (coarse-to-fine)
    for (unsigned int i = 0; i < ctf_levels; i++)
    {
        const unsigned int width = colour_wf.cols();
        s = pow(2.f, int(ctf_levels - (i + 1)));
        cols_i = cols / s;
        rows_i = rows / s;
        level_image = ctf_levels - i + round(log2(width / cols)) - 1;

        //=========================================================================
        //                              Cuda - Begin
        //=========================================================================

        // Cuda allocate memory
        csf_host.allocateMemoryNewLevel(rows_i, cols_i, i, level_image);

        // Cuda copy object to device
        csf_device = ObjectToDevice(&csf_host);

        // Assign zeros to the corresponding variables
        AssignZerosBridge(csf_device);

        // Upsample previous solution
        if (i > 0)
            UpsampleBridge(csf_device);

        // Compute connectivity (Rij)
        RijBridge(csf_device);

        // Compute colour and depth derivatives
        ImageGradientsBridge(csf_device);
        WarpingBridge(csf_device);

        // Compute mu_uv and step sizes for the primal-dual algorithm
        MuAndStepSizesBridge(csf_device);

        // Primal-Dual solver
        for (num_iter = 0; num_iter < num_max_iter[i]; num_iter++)
        {
            GradientBridge(csf_device);
            DualVariablesBridge(csf_device);
            DivergenceBridge(csf_device);
            PrimalVariablesBridge(csf_device);
        }

        // Filter solution
        FilterBridge(csf_device);

        // Compute the motion field
        MotionFieldBridge(csf_device);

        // BridgeBack
        BridgeBack(&csf_host, csf_device);

        // Free variables of variables associated to this level
        csf_host.freeLevelVariables();

        // Copy motion field and images to CPU
        csf_host.copyAllSolutions(dx[ctf_levels - i - 1].data(), dy[ctf_levels - i - 1].data(), dz[ctf_levels - i - 1].data(),
                                  depth[level_image].data(), depth_old[level_image].data(), colour[level_image].data(), colour_old[level_image].data(),
                                  xx[level_image].data(), xx_old[level_image].data(), yy[level_image].data(), yy_old[level_image].data());

        //=========================================================================
        //                              Cuda - end
        //=========================================================================
    }
}

void PD_flow::process_frame(cv::Mat &rgb_image, cv::Mat &depth_image)
{
    // Verifica que las imágenes no estén vacías
    if (rgb_image.empty() || depth_image.empty())
    {
        throw std::invalid_argument("Las imágenes no deben estar vacías.");
    }

    // Convierte la imagen RGB de cv::Mat a Eigen::MatrixXf

    for (int i = 0; i < rgb_image.rows; ++i)
    {
        for (int j = 0; j < rgb_image.cols; ++j)
        {
            // Supongamos que quieres tomar solo el canal rojo de la imagen RGB
            colour_wf(i, j) = static_cast<float>(rgb_image.at<cv::Vec3b>(i, j)[2]); // Canal rojo
        }
    }

    // Convierte la imagen de profundidad de cv::Mat a Eigen::MatrixXf

    for (int i = 0; i < depth_image.rows; ++i)
    {
        for (int j = 0; j < depth_image.cols; ++j)
        {
            depth_wf(i, j) = static_cast<float>(depth_image.at<uchar>(i, j));
        }
    }
}

void PD_flow::process_frame2(cv::Mat &rgb_image, cv::Mat &depth_image)
{

    // Convert the RGB image to grayscale
    cv::Mat gray_image;
    cv::cvtColor(rgb_image, gray_image, cv::COLOR_BGR2GRAY);

    // Ensure the gray_image is of type CV_32FC1 (float, 1 channel)
    if (gray_image.type() != CV_32FC1)
    {
        gray_image.convertTo(gray_image, CV_32FC1);
    }

    // Ensure the depth_image is of type CV_32FC1 (float, 1 channel)
    if (depth_image.type() != CV_32FC1)
    {
        depth_image.convertTo(depth_image, CV_32FC1);
    }

    // Convert the grayscale image to Eigen matrix
    cv::cv2eigen(gray_image, colour_wf);

    // Convert the depth image to Eigen matrix
    cv::cv2eigen(depth_image, depth_wf);
}

void PD_flow::initializeCUDA()
{
    // Read parameters
    csf_host.readParameters(rows, cols, lambda_i, lambda_d, mu, g_mask, ctf_levels, cam_mode, fovh, fovv);

    // Allocate memory
    csf_host.allocateDevMemory();
}

void PD_flow::freeGPUMemory()
{
    csf_host.freeDeviceMemory();
}

void PD_flow::initializePDFlow()
{
    initializeCUDA();
}

void PD_flow::updateScene()
{
    // Crear imágenes para mostrar el campo de movimiento, color y profundidad
    cv::Mat motion_field = cv::Mat::zeros(rows, cols, CV_8UC3);
    cv::Mat color_image(rows, cols, CV_8UC3);
    cv::Mat depth_image(rows, cols, CV_8UC1);

    const unsigned int repr_level = round(log2(colour_wf.cols() / cols));
    for (unsigned int v = 0; v < rows; v++)
    {
        for (unsigned int u = 0; u < cols; u++)
        {
            // Obtener la profundidad y los desplazamientos de cada punto
            float depth_value = depth[repr_level](v, u);
            if (depth_value > 0.0f)
            {
                // Escalar los valores de desplazamiento para visualizarlos mejor
                float dx_scaled = dx[repr_level](v, u);
                float dy_scaled = dy[repr_level](v, u);

                // Calcular el módulo del vector de desplazamiento
                float displacement_magnitude = sqrt(dx_scaled * dx_scaled + dy_scaled * dy_scaled);

                // Normalizar el valor de desplazamiento para el rango de color (0 a 255)
                float max_displacement = 50.0f; // Ajusta este valor según la escala esperada de desplazamiento
                float normalized_magnitude = std::min(displacement_magnitude / max_displacement, 1.0f);

                // Calcular el color basado en la magnitud del desplazamiento (azul para pequeño, rojo para grande)
                cv::Scalar color(255 * (1 - normalized_magnitude), 0, 255 * (normalized_magnitude));

                // Dibujar la línea que representa el vector de movimiento
                cv::Point2f start_point(u, v);
                cv::Point2f end_point(u + dx_scaled, v + dy_scaled);

                // Dibujar el vector en la imagen
                cv::arrowedLine(motion_field, start_point, end_point, color, 1, cv::LINE_AA);

                // Convertir valores de color y profundidad a formatos adecuados para visualización
            }
            color_image.at<cv::Vec3b>(v, u) = cv::Vec3b(colour_wf(v, u), colour_wf(v, u), colour_wf(v, u));

            depth_image.at<uint8_t>(v, u) = static_cast<uint8_t>(depth_value * 255); // Normalizar la profundidad para visualización
        }
    }

    // Mostrar la imagen del campo de movimiento
    cv::imshow("Motion Field", motion_field);

    // Mostrar la imagen de color
    cv::imshow("Color Image", color_image);

    // Mostrar la imagen de profundidad
    cv::imshow("Depth Image", depth_image);

    cv::waitKey(1); // Espera breve para actualizar las ventanas
}

pd_flow_msgs::msg::FlowField PD_flow::convertToFlowFieldMsg()
{
    // Crear un mensaje MotionField
    pd_flow_msgs::msg::FlowField msg;

    // Convertir las matrices Eigen a arrays unidimensionales y asignarlos al mensaje
    for (const auto &matrix : dx)
    {
        for (int i = 0; i < matrix.size(); ++i)
        {
            msg.dx.push_back(matrix(i));
        }
    }

    for (const auto &matrix : dy)
    {
        for (int i = 0; i < matrix.size(); ++i)
        {
            msg.dy.push_back(matrix(i));
        }
    }

    for (const auto &matrix : dz)
    {
        for (int i = 0; i < matrix.size(); ++i)
        {
            msg.dz.push_back(matrix(i));
        }
    }

    return msg;
}

void PD_flow::processPointCloud(std::vector<cv::Point3f> &points, std::vector<cv::Point3f> &vectors)
{
    // Compute the representation level
    const unsigned int repr_level = std::round(std::log2(colour_wf.cols() / cols));

    // Ensure the depth arrays are properly indexed
    const Eigen::MatrixXf &depth_current = depth[repr_level];
    const Eigen::MatrixXf &xx_current = xx[repr_level];
    const Eigen::MatrixXf &yy_current = yy[repr_level];

    // Prepare the point cloud
    points.clear();
    vectors.clear();
    for (unsigned int v = 0; v < rows; ++v)
    {
        for (unsigned int u = 0; u < cols; ++u)
        {
            float depth_value = depth_current(v, u);
            if (depth_value > 0.1f)
            {
                float dx_scaled = dx[repr_level](v, u);
                float dy_scaled = dy[repr_level](v, u);
                float dz_scaled = dz[repr_level](v, u);

                points.emplace_back(xx_current(v, u), yy_current(v, u), depth_current(v, u));
                vectors.emplace_back(dx_scaled, dy_scaled, dz_scaled);
            }
        }
    }
}

sensor_msgs::msg::PointCloud2 PD_flow::createPointCloud()
{
    auto pointcloud_msg = sensor_msgs::msg::PointCloud2();

    sensor_msgs::PointCloud2Modifier modifier(pointcloud_msg);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(rows * cols);

    sensor_msgs::PointCloud2Iterator<float> iter_x(pointcloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(pointcloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(pointcloud_msg, "z");

    // Calcular el nivel de representación
    const unsigned int repr_level = std::round(std::log2(colour_wf.cols() / cols));

    // Asegurar que las matrices de profundidad están indexadas correctamente
    const Eigen::MatrixXf &depth_current = depth[repr_level];
    const Eigen::MatrixXf &xx_current = xx[repr_level];
    const Eigen::MatrixXf &yy_current = yy[repr_level];

    for (unsigned int v = 0; v < rows; ++v)
    {
        for (unsigned int u = 0; u < cols; ++u, ++iter_x, ++iter_y, ++iter_z)
        {
            *iter_x = depth_current(v, u);
            *iter_y = xx_current(v, u);
            *iter_z = yy_current(v, u);
        }
    }

    return pointcloud_msg;
}

visualization_msgs::msg::MarkerArray PD_flow::createVectorField(const builtin_interfaces::msg::Time &current_time)
{
    // Placeholder data - replace with your actual data
    visualization_msgs::msg::MarkerArray marker_array;
    int id = 0;

    // Calcular el nivel de representación
    const unsigned int repr_level = std::round(std::log2(colour_wf.cols() / cols));

    // Asegurar que las matrices de profundidad están indexadas correctamente
    const Eigen::MatrixXf &depth_current = depth[repr_level];
    const Eigen::MatrixXf &xx_current = xx[repr_level];
    const Eigen::MatrixXf &yy_current = yy[repr_level];
    const Eigen::MatrixXf &dx_current = dx[repr_level];
    const Eigen::MatrixXf &dy_current = dy[repr_level];
    const Eigen::MatrixXf &dz_current = dz[repr_level];

    for (unsigned int v = 0; v < rows; ++v)
    {
        for (unsigned int u = 0; u < cols; ++u)
        {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = current_time;
            marker.ns = "vector_field";
            marker.id = id++;
            marker.type = visualization_msgs::msg::Marker::ARROW;
            marker.action = visualization_msgs::msg::Marker::ADD;

            marker.pose.position.x = depth_current(v, u);
            marker.pose.position.y = xx_current(v, u);
            marker.pose.position.z = yy_current(v, u);

            tf2::Quaternion quat;
            float angle = atan2(dy_current(v, u), dx_current(v, u));
            quat.setRPY(0.0, 0.0, angle);
            marker.pose.orientation.x = quat.x();
            marker.pose.orientation.y = quat.y();
            marker.pose.orientation.z = quat.z();
            marker.pose.orientation.w = quat.w();

            float length = sqrt(dx_current(v, u) * dx_current(v, u) +
                                dy_current(v, u) * dy_current(v, u) +
                                dz_current(v, u) * dz_current(v, u));
            marker.scale.x = length;
            marker.scale.y = 0.1;
            marker.scale.z = 0.1;

            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;

            marker_array.markers.push_back(marker);
        }
    }

    return marker_array;
}