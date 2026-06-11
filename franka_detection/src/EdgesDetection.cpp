#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <franka_msgs/msg/edges.hpp>

// ViSP
#include <visp3/gui/vpDisplayOpenCV.h>

#include <visp3/core/vpImage.h>
#include <visp3/core/vpImageConvert.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/mbt/vpMbEdgeTracker.h>
#include <visp3/core/vpException.h>
#include <visp3/sensor/vpRealSense2.h>






#include "tf2/exceptions.h"
#include "tf2_ros/message_filter.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

#include "ament_index_cpp/get_package_share_directory.hpp"

class EdgesDetectionNode : public rclcpp::Node
{
    public:
    EdgesDetectionNode()
    : Node("edges_detection_node")
    {        

        model_path_ = ament_index_cpp::get_package_share_directory("franka_detection")
                  + "/models/box.cao";


        rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/right_camera/image", 5,
        std::bind(&EdgesDetectionNode::rgb_callback, this, std::placeholders::_1));

        
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/right_camera/depth_image", 5,
        std::bind(&EdgesDetectionNode::depth_callback, this, std::placeholders::_1));

        
        info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/right_camera/camera_info", 5,
        std::bind(&EdgesDetectionNode::camera_info_callback, this, std::placeholders::_1));

    
        edges_pub_ = this->create_publisher<franka_msgs::msg::Edges>("/edges", 1);
        debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/edges_detection/debug_image", 1);

        
        timer_ = this->create_wall_timer(
        std::chrono::milliseconds(33),
        std::bind(&EdgesDetectionNode::on_timer, this));     
        
        
        tf_buffer_ =std::make_shared<tf2_ros::Buffer>(this->get_clock(),std::chrono::seconds(10));
        tf_listener_ =std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    }

  private:


    void rgb_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        rgb_image_ = cv_ptr->image;  // cv::Mat BGR
             
        
        visp::vpImageConvert::convert(rgb_image_, I_);  
        visp::vpImageConvert::convert(rgb_image_, I_color_);  

        if (!display_ && I_color_.getSize() > 0) {
        display_ = new visp::vpDisplayOpenCV();
        display_->init(I_color_, -1, -1, "Debug Window");
    }
        
        is_image_ = true;
    }

    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        auto cv_ptr = cv_bridge::toCvCopy(msg, "32FC1");
        depth_image_ = cv_ptr->image;  // cv::Mat float32 in metri
    }

    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        //Prelevo i dati della camera
        if (camera_ready_) return;

        double fx=msg->k[0];
        double fy=msg->k[4];
        double cx=msg->k[2];
        double cy=msg->k[5];
                    
        cam_.initPersProjWithoutDistortion(fx, fy, cx, cy);
        camera_ready_ = true;

       

        //Carico modello dell'oggetto .cao e poi i parametri per le detection
        tracker_.loadModel(model_path_);
        tracker_.setCameraParameters(cam_);
        
        
        tracker_.setAngleAppear(visp::vpMath::rad(86)); //Da tenere altini per massimizzare la probabilità di vedere facce
        tracker_.setAngleDisappear(visp::vpMath::rad(89));
        tracker_.setNearClippingDistance(0.1);
        tracker_.setFarClippingDistance(100.0);
        tracker_.setClipping(tracker_.getClipping() | visp::vpMbtPolygon::FOV_CLIPPING);

        // visp::vpKltOpencv klt_settings;
        // klt_settings.setMaxFeatures(300);
        // klt_settings.setWindowSize(5);
        // klt_settings.setQuality(0.015);
        // klt_settings.setMinDistance(8);
        // klt_settings.setHarrisFreeParameter(0.01);
        // klt_settings.setBlockSize(3);
        // klt_settings.setPyramidLevels(3);
        // tracker_.setKltOpencv(klt_settings);

        visp::vpMe me;
        me.setMaskSize(3);
        me.setMaskNumber(180);
        me.setRange(20); //quanto
        me.setThreshold(2000); //Modulo del gradiente
        me.setMu1(0.5); me.setMu2(0.5);
        me.setSampleStep(2);
        me.setNbTotalSample(250);
        tracker_.setMovingEdge(me);

    
    }

    void on_timer()
    {
        //Aspetto che la camera sia pronta e l'immagine ricevuta
        if (!camera_ready_ || !is_image_) {
          
          return;
        }
        //Inizializzo il tracker
        if (!tracker_initialized_) {
            if (!setup_visp_tracker()) return; 
        }

        

        try {
             tracker_.track(I_);  
            tracker_.getPose(cMo_);
                    
        } catch (const visp::vpException & e) {
            RCLCPP_WARN(this->get_logger(), "Tracking perso: %s", e.what());
            tracker_initialized_ = false;
            return;
        }
       
      
        if (display_) {
            // Prepara il frame per il disegno
            visp::vpDisplay::display(I_color_);
            
            // Disegna il modello (linee rosse) e il frame XYZ
            tracker_.display(I_color_, cMo_, cam_, visp::vpColor::red, 2);
            visp::vpDisplay::displayFrame(I_color_, cMo_, cam_, 0.025, visp::vpColor::none, 3);
            
            // Forza ViSP a renderizzare il buffer
            visp::vpDisplay::flush(I_color_);
            
            // 3. SOVRASCRIVI I PIXEL DI I_color_ CON IL DISEGNO
            visp::vpDisplay::getImage(I_color_, I_color_);
        }

        // 4. CONVERSIONE E PUBBLICAZIONE ROS 2
        cv::Mat debug_mat;
        visp::vpImageConvert::convert(I_color_, debug_mat);

        auto debug_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", debug_mat).toImageMsg();
        debug_msg->header.stamp = this->now();
        debug_image_pub_->publish(*debug_msg);

    }


    bool setup_visp_tracker()
    {
       try {
            
            //Necessito di una trasformazione da camera link optical a box come stima inziiale
            //da dare al traker
            auto Tco = tf_buffer_->lookupTransform("right_fr3_camera_link_optical", "box", tf2::TimePointZero);
            
            visp::vpTranslationVector transl_vec{
                Tco.transform.translation.x, Tco.transform.translation.y, Tco.transform.translation.z};
            
            visp::vpQuaternionVector quat_vec{
                Tco.transform.rotation.x, Tco.transform.rotation.y, Tco.transform.rotation.z, Tco.transform.rotation.w};

            cMo_.buildFrom(transl_vec, quat_vec);
            tracker_.initFromPose(I_, cMo_);
            
            tracker_initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "Tracker ViSP inizializzato con successo!");
            return true;

        } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                "In attesa del frame TF dell'oggetto: %s", ex.what());
            return false;
        }
      
               
    }



    //Membri ros2
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    rclcpp::Publisher<franka_msgs::msg::Edges>::SharedPtr edges_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;


    rclcpp::TimerBase::SharedPtr timer_;

    //Membri visp
    visp::vpCameraParameters cam_;
  
    visp::vpMbEdgeTracker tracker_;
    std::string model_path_;
    visp::vpImage<unsigned char> I_;
    visp::vpImage<visp::vpRGBa> I_color_;
    visp::vpHomogeneousMatrix cMo_;

    //Flag per sincro
    bool camera_ready_=false;
    bool is_image_=false;
    bool tracker_initialized_=false;

    //OpenCV
    cv::Mat rgb_image_;
    cv::Mat depth_image_;

    //Per visualizzare
    visp::vpDisplayOpenCV* display_ {nullptr};
    
  

   


};


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EdgesDetectionNode>());
  rclcpp::shutdown();
  return 0;
}