#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <path_searching/dyn_a_star.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>

using PointT = pcl::PointXYZI;
using CloudT = pcl::PointCloud<PointT>;

class GlobalAstarPlanner {
 public:
  GlobalAstarPlanner() : nh_(), pnh_("~") {
    pnh_.param<std::string>("map_path", map_path_, "");
    pnh_.param<std::string>("odom_topic", odom_topic_, "/Odometry");
    pnh_.param<std::string>("goal_topic", goal_topic_, "/move_base_simple/goal");
    pnh_.param<std::string>("path_topic", path_topic_, "/global_path");
    pnh_.param<std::string>("frame_id", frame_id_, "world");
    pnh_.param<double>("resolution", resolution_, 0.25);
    pnh_.param<double>("search_step", search_step_, 0.5);
    pnh_.param<double>("map_margin", map_margin_, 2.0);
    pnh_.param<double>("obstacles_inflation", obstacles_inflation_, 0.45);
    pnh_.param<double>("pcd_leaf_size", pcd_leaf_size_, 0.25);
    pnh_.param<bool>("use_goal_z", use_goal_z_, false);
    pnh_.param<double>("default_goal_z", default_goal_z_, 1.2);
    pnh_.param<double>("replan_period", replan_period_, 1.0);

    loadStaticMap();

    odom_sub_ = nh_.subscribe(odom_topic_, 1, &GlobalAstarPlanner::odomCb, this);
    goal_sub_ = nh_.subscribe(goal_topic_, 1, &GlobalAstarPlanner::goalCb, this);
    path_pub_ = nh_.advertise<nav_msgs::Path>(path_topic_, 1, true);
    replan_timer_ = nh_.createTimer(ros::Duration(replan_period_), &GlobalAstarPlanner::timerCb, this);
    ROS_INFO("global_astar_planner ready: map=%s path=%s", map_path_.c_str(), path_topic_.c_str());
  }

 private:
  void loadStaticMap() {
    if (map_path_.empty()) {
      throw std::runtime_error("~map_path is required");
    }
    CloudT::Ptr raw(new CloudT);
    if (pcl::io::loadPCDFile<PointT>(map_path_, *raw) != 0 || raw->empty()) {
      throw std::runtime_error("failed to load non-empty PCD map: " + map_path_);
    }

    CloudT::Ptr cloud(new CloudT);
    if (pcd_leaf_size_ > 0.0) {
      pcl::VoxelGrid<PointT> voxel;
      voxel.setLeafSize(pcd_leaf_size_, pcd_leaf_size_, pcd_leaf_size_);
      voxel.setInputCloud(raw);
      voxel.filter(*cloud);
    } else {
      cloud = raw;
    }

    Eigen::Vector3d min_pt(
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity());
    Eigen::Vector3d max_pt(
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity());
    for (const auto& p : cloud->points) {
      min_pt = min_pt.cwiseMin(Eigen::Vector3d(p.x, p.y, p.z));
      max_pt = max_pt.cwiseMax(Eigen::Vector3d(p.x, p.y, p.z));
    }
    min_pt.array() -= map_margin_;
    max_pt.array() += map_margin_;
    if (max_pt.z() - min_pt.z() < 2.0) max_pt.z() = min_pt.z() + 2.0;

    const Eigen::Vector3d size = max_pt - min_pt;
    pnh_.setParam("grid_map/resolution", resolution_);
    pnh_.setParam("grid_map/map_size_x", size.x());
    pnh_.setParam("grid_map/map_size_y", size.y());
    pnh_.setParam("grid_map/map_size_z", size.z());
    pnh_.setParam("grid_map/origin_x", min_pt.x());
    pnh_.setParam("grid_map/origin_y", min_pt.y());
    pnh_.setParam("grid_map/origin_z", min_pt.z());
    pnh_.setParam("grid_map/local_update_range_x", size.x());
    pnh_.setParam("grid_map/local_update_range_y", size.y());
    pnh_.setParam("grid_map/local_update_range_z", size.z());
    pnh_.setParam("grid_map/obstacles_inflation", obstacles_inflation_);
    pnh_.setParam("grid_map/frame_id", frame_id_);
    pnh_.setParam("grid_map/pose_type", GridMap::ODOMETRY);
    pnh_.setParam("grid_map/ground_height", min_pt.z());
    pnh_.setParam("grid_map/skip_pixel", 4);

    grid_map_.reset(new GridMap);
    grid_map_->initMap(pnh_);

    const int inflate_steps = std::max(0, static_cast<int>(std::ceil(obstacles_inflation_ / resolution_)));
    for (const auto& p : cloud->points) {
      for (int dx = -inflate_steps; dx <= inflate_steps; ++dx) {
        for (int dy = -inflate_steps; dy <= inflate_steps; ++dy) {
          for (int dz = -inflate_steps; dz <= inflate_steps; ++dz) {
            Eigen::Vector3d occ(
                p.x + dx * resolution_,
                p.y + dy * resolution_,
                p.z + dz * resolution_);
            grid_map_->setOccupied(occ);
          }
        }
      }
    }

    Eigen::Vector3i pool_size(
        static_cast<int>(std::ceil(size.x() / search_step_)) + 8,
        static_cast<int>(std::ceil(size.y() / search_step_)) + 8,
        static_cast<int>(std::ceil(size.z() / search_step_)) + 8);
    astar_.reset(new AStar);
    astar_->initGridMap(grid_map_, pool_size);
    ROS_INFO("A* static map loaded: raw=%zu filtered=%zu origin=[%.2f %.2f %.2f] size=[%.2f %.2f %.2f]",
             raw->size(), cloud->size(), min_pt.x(), min_pt.y(), min_pt.z(), size.x(), size.y(), size.z());
  }

  void odomCb(const nav_msgs::OdometryConstPtr& msg) {
    odom_ = *msg;
    has_odom_ = true;
  }

  void goalCb(const geometry_msgs::PoseStampedConstPtr& msg) {
    goal_ = *msg;
    has_goal_ = true;
    plan();
  }

  void timerCb(const ros::TimerEvent&) {
    if (has_goal_) plan();
  }

  void plan() {
    if (!has_odom_ || !has_goal_) return;
    Eigen::Vector3d start(
        odom_.pose.pose.position.x,
        odom_.pose.pose.position.y,
        odom_.pose.pose.position.z);
    Eigen::Vector3d goal(
        goal_.pose.position.x,
        goal_.pose.position.y,
        use_goal_z_ ? goal_.pose.position.z : default_goal_z_);

    if (!grid_map_->isInMap(start) || !grid_map_->isInMap(goal)) {
      ROS_WARN_THROTTLE(1.0, "A* start or goal outside static map");
      return;
    }
    if (!astar_->AstarSearch(search_step_, start, goal)) {
      ROS_WARN_THROTTLE(1.0, "A* failed from [%.2f %.2f %.2f] to [%.2f %.2f %.2f]",
                        start.x(), start.y(), start.z(), goal.x(), goal.y(), goal.z());
      return;
    }
    const std::vector<Eigen::Vector3d> pts = astar_->getPath();
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = frame_id_;
    for (const auto& pt : pts) {
      geometry_msgs::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = pt.x();
      pose.pose.position.y = pt.y();
      pose.pose.position.z = pt.z();
      pose.pose.orientation.w = 1.0;
      path.poses.push_back(pose);
    }
    path_pub_.publish(path);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber odom_sub_;
  ros::Subscriber goal_sub_;
  ros::Publisher path_pub_;
  ros::Timer replan_timer_;

  std::string map_path_;
  std::string odom_topic_;
  std::string goal_topic_;
  std::string path_topic_;
  std::string frame_id_;
  double resolution_;
  double search_step_;
  double map_margin_;
  double obstacles_inflation_;
  double pcd_leaf_size_;
  bool use_goal_z_;
  double default_goal_z_;
  double replan_period_;

  GridMap::Ptr grid_map_;
  AStar::Ptr astar_;
  nav_msgs::Odometry odom_;
  geometry_msgs::PoseStamped goal_;
  bool has_odom_ = false;
  bool has_goal_ = false;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "global_astar_planner");
  try {
    GlobalAstarPlanner node;
    ros::spin();
  } catch (const std::exception& exc) {
    ROS_FATAL("%s", exc.what());
    return 1;
  }
  return 0;
}
