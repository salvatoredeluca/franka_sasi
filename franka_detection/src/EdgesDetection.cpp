#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <franka_msgs/msg/line.hpp>
#include <franka_msgs/msg/edges.hpp>

// OpenCV
#include <opencv2/opencv.hpp>

#include "ament_index_cpp/get_package_share_directory.hpp"

class EdgesDetectionNode : public rclcpp::Node
{
public:
    EdgesDetectionNode()
    : Node("edges_detection_node")
    {        
        // Puoi tenere questo per usi futuri se serve, altrimenti puoi rimuoverlo
        model_path_ = ament_index_cpp::get_package_share_directory("franka_detection")
                  + "/models/box.cao";

        //For simulation
        // rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        //     "/left/camera_left/color/image_raw", 5,
        //     std::bind(&EdgesDetectionNode::rgb_callback, this, std::placeholders::_1));     
        // depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        //     "/left/camera_left/color/image_raw/compressedDepth", 5,
        //     std::bind(&EdgesDetectionNode::depth_callback, this, std::placeholders::_1));       
        // info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        //     "/left/camera_left/color/camera_info", 5,
        //     std::bind(&EdgesDetectionNode::camera_info_callback, this, std::placeholders::_1));

        //In real
        rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/left/camera_left/color/image_raw", 5,
            std::bind(&EdgesDetectionNode::rgb_callback, this, std::placeholders::_1));     
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/left/camera_left/color/image_raw/compressedDepth", 5,
            std::bind(&EdgesDetectionNode::depth_callback, this, std::placeholders::_1));       
        info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/left/camera_left/color/camera_info", 5,
            std::bind(&EdgesDetectionNode::camera_info_callback, this, std::placeholders::_1));
    
        edges_pub_ = this->create_publisher<franka_msgs::msg::Edges>("/edges", 1);
        
        // Publisher per l'immagine di debug
        debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/edges_detection/debug_image", 1);
        
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33),
            std::bind(&EdgesDetectionNode::on_timer, this));
    }

private:

    void rgb_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        rgb_image_ = cv_ptr->image;  
        is_image_ = true;
    }

    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Se usi profondità in metri, 32FC1 è corretto. 
        // Se usi millimetri (es. RealSense standard), potrebbe essere 16UC1.
        auto cv_ptr = cv_bridge::toCvCopy(msg, "32FC1"); 
        depth_image_ = cv_ptr->image;  
    }

    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (camera_ready_) return;

        fx_ = msg->k[0]; fy_ = msg->k[4];
        cx_ = msg->k[2]; cy_ = msg->k[5];
                    
        camera_ready_ = true;
    }


    // Prendo la profonità del centro
    float get_median_depth(const cv::Mat& depth, int u, int v, int window_size = 5) {
        std::vector<float> depths;
        int half_w = window_size / 2;
        for (int dy = -half_w; dy <= half_w; ++dy) {
            for (int dx = -half_w; dx <= half_w; ++dx) {//assegno la profondità ad ognuno dei 25 pixel intorno al centro
                int nx = u + dx;
                int ny = v + dy;
                // Controlla che non usciamo dai bordi dell'immagine
                if (nx >= 0 && nx < depth.cols && ny >= 0 && ny < depth.rows) {
                    float d = depth.at<float>(ny, nx);
                    // Accetta solo profondità valide (es. > 10cm)
                    if (d > 0.1f && !std::isnan(d)) { 
                        depths.push_back(d);//creo la lista di profondità dei pixel intorno al centro
                    }
                }
            }
        }
        if (depths.empty()) return 0.0f;
        std::sort(depths.begin(), depths.end());
        return depths[depths.size() / 2]; // Ritorna la mediana
    }

    void on_timer()
    {
        franka_msgs::msg::Edges edges_msg;
        if (!camera_ready_ || !is_image_) {
            return;
        }

        line_vector_.clear();

        // 1. Creiamo una copia dell'immagine RGB originale su cui disegnare
        cv::Mat debug_image = rgb_image_.clone();


       
       
        cv::Mat hsv, mask;
        cv::cvtColor(rgb_image_, hsv, cv::COLOR_BGR2HSV);

       
        cv::Scalar lower(35, 180, 200); // H min, S min (alta), V min (alto)
        cv::Scalar upper(85, 255, 255); // H max, S max, V max
        
        cv::inRange(hsv, lower, upper, mask);
        //Toglie rumore
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Sicurezza: procedi solo se ha trovato almeno una macchia di colore
        if (!contours.empty()) {
            
            // Trovi il più grande (nota: al momento non lo stai usando nel resto del codice, 
            // ma lo lascio esattamente come lo hai scritto tu).
            auto biggest = *std::max_element(contours.begin(), contours.end(),
                [](auto& a, auto& b){ return cv::contourArea(a) < cv::contourArea(b); });

            for (int i=0; i<contours.size(); i++)
            {
                // 1. Filtro dimensione: scartiamo il rumore e prendiamo solo oggetti grandi
                if (cv::contourArea(contours[i]) > 500.0) 
                {
                    cv::Rect bbox = cv::boundingRect(contours.at(i));

                    // 2. FILTRO GEOMETRICO: Calcoliamo l'Aspect Ratio
                    float aspect_ratio = (float)bbox.width / bbox.height;

                  
                        // Disegnamo SOLO questo contorno (nota l'indice 'i' al posto di '-1')
                        cv::drawContours(debug_image, contours, i, cv::Scalar(0, 255, 0), 2);
                        
                        // Disegniamo il rettangolo rosso
                        cv::rectangle(debug_image, bbox, cv::Scalar(0, 0, 255), 2);

                        // --- INIZIO DELLA TUA LOGICA 3D ---
                        cv::Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
                        
                        float z = get_median_depth(depth_image_, center.x, center.y);

                        auto deproject = [&](int u, int v, float depth) {
                                geometry_msgs::msg::Point pt;
                                pt.x = (u - cx_) * depth / fx_;
                                pt.y = (v - cy_) * depth / fy_;
                                pt.z = depth;
                                return pt;
                            };
                            
                        auto pt_tl = deproject(bbox.x, bbox.y, z);                               // Top-Left
                        auto pt_tr = deproject(bbox.x + bbox.width, bbox.y, z);                  // Top-Right
                        auto pt_br = deproject(bbox.x + bbox.width, bbox.y + bbox.height, z);    // Bottom-Right
                        auto pt_bl = deproject(bbox.x, bbox.y + bbox.height, z);

                        franka_msgs::msg::Line l_top, l_right, l_bottom, l_left;
                        l_top.point_a = pt_tl;    l_top.point_b = pt_tr;
                        l_right.point_a = pt_tr;  l_right.point_b = pt_br;
                        l_bottom.point_a = pt_br; l_bottom.point_b = pt_bl;
                        l_left.point_a = pt_bl;   l_left.point_b = pt_tl;

                        // Salviamo le linee della BORRACCIA nel vettore
                        line_vector_.insert(line_vector_.end(), {l_top, l_right, l_bottom, l_left});
                        // --- FINE DELLA TUA LOGICA 3D ---
                    
                }
            }
        }
        
        edges_msg.lines = line_vector_; // <-- Sintassi corretta
        edges_msg.header.stamp = this->now();
        edges_msg.header.frame_id = "left_fr3_camera_link_optical"; 
                
        edges_pub_->publish(edges_msg);
                    
      
        // 4. Converti l'immagine modificata in un messaggio ROS e pubblicala
        auto debug_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", debug_image).toImageMsg();
        debug_msg->header.stamp = this->now();
        
        debug_msg->header.frame_id = "left_fr3_camera_link_optical"; 
        debug_image_pub_->publish(*debug_msg);
    }

    // ── Membri ROS2 ──────────────────────────────────────────────────────────────
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    rclcpp::Publisher<franka_msgs::msg::Edges>::SharedPtr edges_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::string model_path_;

    std::vector<franka_msgs::msg::Line> line_vector_;
    
    // ── Flag sincronizzazione ────────────────────────────────────────────────────
    bool camera_ready_        = false;
    bool is_image_            = false;
    bool tracker_initialized_ = false;

    // ── OpenCV & Display ─────────────────────────────────────────────────────────
    cv::Mat rgb_image_;
    cv::Mat depth_image_;

    double fx_;
    double fy_;
    double cx_;
    double cy_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EdgesDetectionNode>());
    rclcpp::shutdown();
    return 0;
}