#include <cmath>
#include <string>

#include <Eigen/Dense>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

using PointT = pcl::PointXYZI;
using CloudT = pcl::PointCloud<PointT>;

class PcdLocalization {
 public:
  PcdLocalization() : nh_(), pnh_("~") {
    pnh_.param<std::string>("map_path", map_path_, "");
    pnh_.param<std::string>("input_topic", input_topic_, "/livox/lidar2");
    pnh_.param<std::string>("input_type", input_type_, "pointcloud2");
    pnh_.param<std::string>("odom_topic", odom_topic_, "/Odometry");
    pnh_.param<std::string>("cloud_topic", cloud_topic_, "/cloud_registered");
    pnh_.param<std::string>("frame_id", frame_id_, "world");
    pnh_.param<std::string>("child_frame_id", child_frame_id_, "base_link");
    pnh_.param<double>("map_leaf_size", map_leaf_size_, 0.25);
    pnh_.param<double>("scan_leaf_size", scan_leaf_size_, 0.2);
    pnh_.param<double>("ndt_resolution", ndt_resolution_, 1.0);
    pnh_.param<double>("ndt_step_size", ndt_step_size_, 0.1);
    pnh_.param<double>("ndt_trans_eps", ndt_trans_eps_, 0.01);
    pnh_.param<int>("ndt_max_iter", ndt_max_iter_, 30);
    pnh_.param<double>("icp_max_corr", icp_max_corr_, 1.0);
    pnh_.param<double>("icp_trans_eps", icp_trans_eps_, 0.001);
    pnh_.param<int>("icp_max_iter", icp_max_iter_, 20);
    pnh_.param<double>("initial_x", initial_x_, 0.0);
    pnh_.param<double>("initial_y", initial_y_, 0.0);
    pnh_.param<double>("initial_z", initial_z_, 0.0);
    pnh_.param<double>("initial_yaw", initial_yaw_, 0.0);

    pose_ = yawPose(initial_x_, initial_y_, initial_z_, initial_yaw_);
    loadMap();

    odom_pub_ = nh_.advertise<nav_msgs::Odometry>(odom_topic_, 10);
    cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(cloud_topic_, 5);
    initialpose_sub_ = nh_.subscribe("/initialpose", 1, &PcdLocalization::initialPoseCb, this);
    if (input_type_ == "pointcloud") {
      pointcloud_sub_ = nh_.subscribe(input_topic_, 1, &PcdLocalization::pointCloudCb, this);
    } else {
      pointcloud2_sub_ = nh_.subscribe(input_topic_, 1, &PcdLocalization::pointCloud2Cb, this);
    }

    ROS_INFO("pcd_localization ready: map=%s input=%s type=%s odom=%s cloud=%s",
             map_path_.c_str(), input_topic_.c_str(), input_type_.c_str(),
             odom_topic_.c_str(), cloud_topic_.c_str());
  }

 private:
  static Eigen::Matrix4f yawPose(double x, double y, double z, double yaw) {
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    const float c = std::cos(yaw);
    const float s = std::sin(yaw);
    pose(0, 0) = c;
    pose(0, 1) = -s;
    pose(1, 0) = s;
    pose(1, 1) = c;
    pose(0, 3) = x;
    pose(1, 3) = y;
    pose(2, 3) = z;
    return pose;
  }

  void loadMap() {
    if (map_path_.empty()) {
      throw std::runtime_error("~map_path is required");
    }
    CloudT::Ptr raw(new CloudT);
    if (pcl::io::loadPCDFile<PointT>(map_path_, *raw) != 0 || raw->empty()) {
      throw std::runtime_error("failed to load non-empty PCD map: " + map_path_);
    }
    map_.reset(new CloudT);
    if (map_leaf_size_ > 0.0) {
      pcl::VoxelGrid<PointT> voxel;
      voxel.setLeafSize(map_leaf_size_, map_leaf_size_, map_leaf_size_);
      voxel.setInputCloud(raw);
      voxel.filter(*map_);
    } else {
      map_ = raw;
    }
    ndt_.setInputTarget(map_);
    ndt_.setResolution(ndt_resolution_);
    ndt_.setStepSize(ndt_step_size_);
    ndt_.setTransformationEpsilon(ndt_trans_eps_);
    ndt_.setMaximumIterations(ndt_max_iter_);

    icp_.setInputTarget(map_);
    icp_.setMaxCorrespondenceDistance(icp_max_corr_);
    icp_.setTransformationEpsilon(icp_trans_eps_);
    icp_.setMaximumIterations(icp_max_iter_);
    ROS_INFO("Loaded localization map: raw=%zu filtered=%zu", raw->size(), map_->size());
  }

  void initialPoseCb(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg) {
    Eigen::Quaternionf q(
        msg->pose.pose.orientation.w,
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z);
    pose_.setIdentity();
    pose_.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    pose_(0, 3) = msg->pose.pose.position.x;
    pose_(1, 3) = msg->pose.pose.position.y;
    pose_(2, 3) = msg->pose.pose.position.z;
    initialized_ = true;
    ROS_INFO("Localization initial pose reset from /initialpose");
  }

  void pointCloud2Cb(const sensor_msgs::PointCloud2ConstPtr& msg) {
    CloudT::Ptr cloud(new CloudT);
    pcl::fromROSMsg(*msg, *cloud);
    alignAndPublish(cloud, msg->header.stamp);
  }

  void pointCloudCb(const sensor_msgs::PointCloudConstPtr& msg) {
    CloudT::Ptr cloud(new CloudT);
    cloud->reserve(msg->points.size());
    for (const auto& pt : msg->points) {
      PointT p;
      p.x = pt.x;
      p.y = pt.y;
      p.z = pt.z;
      p.intensity = 0.0f;
      cloud->push_back(p);
    }
    alignAndPublish(cloud, msg->header.stamp);
  }

  void alignAndPublish(const CloudT::Ptr& raw, const ros::Time& stamp) {
    if (!raw || raw->empty()) return;

    CloudT::Ptr scan(new CloudT);
    if (scan_leaf_size_ > 0.0) {
      pcl::VoxelGrid<PointT> voxel;
      voxel.setLeafSize(scan_leaf_size_, scan_leaf_size_, scan_leaf_size_);
      voxel.setInputCloud(raw);
      voxel.filter(*scan);
    } else {
      scan = raw;
    }
    if (scan->empty()) return;

    ndt_.setInputSource(scan);
    CloudT ndt_aligned;
    ndt_.align(ndt_aligned, pose_);
    Eigen::Matrix4f guess = ndt_.hasConverged() ? ndt_.getFinalTransformation() : pose_;

    icp_.setInputSource(scan);
    CloudT aligned;
    icp_.align(aligned, guess);
    if (!icp_.hasConverged()) {
      ROS_WARN_THROTTLE(1.0, "ICP did not converge; keeping previous pose");
      return;
    }
    pose_ = icp_.getFinalTransformation();
    initialized_ = true;

    publishOdom(stamp);
    publishCloud(aligned, stamp);
  }

  void publishOdom(const ros::Time& stamp) {
    Eigen::Matrix3f rot = pose_.block<3, 3>(0, 0);
    Eigen::Quaternionf q(rot);
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp.isZero() ? ros::Time::now() : stamp;
    odom.header.frame_id = frame_id_;
    odom.child_frame_id = child_frame_id_;
    odom.pose.pose.position.x = pose_(0, 3);
    odom.pose.pose.position.y = pose_(1, 3);
    odom.pose.pose.position.z = pose_(2, 3);
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    odom_pub_.publish(odom);
  }

  void publishCloud(const CloudT& aligned, const ros::Time& stamp) {
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(aligned, msg);
    msg.header.stamp = stamp.isZero() ? ros::Time::now() : stamp;
    msg.header.frame_id = frame_id_;
    cloud_pub_.publish(msg);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber pointcloud2_sub_;
  ros::Subscriber pointcloud_sub_;
  ros::Subscriber initialpose_sub_;
  ros::Publisher odom_pub_;
  ros::Publisher cloud_pub_;

  std::string map_path_;
  std::string input_topic_;
  std::string input_type_;
  std::string odom_topic_;
  std::string cloud_topic_;
  std::string frame_id_;
  std::string child_frame_id_;
  double map_leaf_size_;
  double scan_leaf_size_;
  double ndt_resolution_;
  double ndt_step_size_;
  double ndt_trans_eps_;
  int ndt_max_iter_;
  double icp_max_corr_;
  double icp_trans_eps_;
  int icp_max_iter_;
  double initial_x_;
  double initial_y_;
  double initial_z_;
  double initial_yaw_;

  CloudT::Ptr map_;
  pcl::NormalDistributionsTransform<PointT, PointT> ndt_;
  pcl::IterativeClosestPoint<PointT, PointT> icp_;
  Eigen::Matrix4f pose_;
  bool initialized_ = false;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "pcd_localization");
  try {
    PcdLocalization node;
    ros::spin();
  } catch (const std::exception& exc) {
    ROS_FATAL("%s", exc.what());
    return 1;
  }
  return 0;
}
