#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace fs = boost::filesystem;

using PointT = pcl::PointXYZI;
using CloudT = pcl::PointCloud<PointT>;

struct Options {
  std::string input_dir;
  std::string output;
  double voxel_leaf = 0.2;
  bool sor = false;
  int sor_mean_k = 24;
  double sor_stddev = 1.0;
};

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " --input_dir third_party/fast_lio/PCD --output assets/maps/powerplant_local.pcd"
            << " [--voxel_leaf 0.2] [--sor] [--sor_mean_k 24] [--sor_stddev 1.0]\n";
}

bool parseArgs(int argc, char** argv, Options& opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto require_value = [&](std::string* value) -> bool {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << arg << "\n";
        return false;
      }
      *value = argv[++i];
      return true;
    };
    if (arg == "--input_dir") {
      if (!require_value(&opt.input_dir)) return false;
    } else if (arg == "--output") {
      if (!require_value(&opt.output)) return false;
    } else if (arg == "--voxel_leaf") {
      std::string value;
      if (!require_value(&value)) return false;
      opt.voxel_leaf = std::stod(value);
    } else if (arg == "--sor") {
      opt.sor = true;
    } else if (arg == "--sor_mean_k") {
      std::string value;
      if (!require_value(&value)) return false;
      opt.sor_mean_k = std::stoi(value);
    } else if (arg == "--sor_stddev") {
      std::string value;
      if (!require_value(&value)) return false;
      opt.sor_stddev = std::stod(value);
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      return false;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }
  return !opt.input_dir.empty() && !opt.output.empty();
}

int main(int argc, char** argv) {
  Options opt;
  if (!parseArgs(argc, argv, opt)) {
    usage(argv[0]);
    return 2;
  }

  if (!fs::exists(opt.input_dir) || !fs::is_directory(opt.input_dir)) {
    std::cerr << "Input directory does not exist: " << opt.input_dir << "\n";
    return 1;
  }

  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(opt.input_dir)) {
    if (!fs::is_regular_file(entry.path())) continue;
    const std::string name = entry.path().filename().string();
    if (entry.path().extension() == ".pcd" && name.find("scans") == 0) {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    std::cerr << "No scans*.pcd files found in " << opt.input_dir << "\n";
    return 1;
  }

  CloudT::Ptr merged(new CloudT);
  for (const auto& file : files) {
    CloudT cloud;
    if (pcl::io::loadPCDFile<PointT>(file.string(), cloud) != 0) {
      std::cerr << "Failed to read " << file.string() << "\n";
      return 1;
    }
    *merged += cloud;
    std::cout << "Loaded " << file.string() << " points=" << cloud.size() << "\n";
  }

  CloudT::Ptr filtered(new CloudT);
  if (opt.voxel_leaf > 0.0) {
    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize(opt.voxel_leaf, opt.voxel_leaf, opt.voxel_leaf);
    voxel.setInputCloud(merged);
    voxel.filter(*filtered);
  } else {
    filtered = merged;
  }

  CloudT::Ptr denoised(new CloudT);
  if (opt.sor) {
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(filtered);
    sor.setMeanK(opt.sor_mean_k);
    sor.setStddevMulThresh(opt.sor_stddev);
    sor.filter(*denoised);
  } else {
    denoised = filtered;
  }

  fs::path out(opt.output);
  if (!out.parent_path().empty()) fs::create_directories(out.parent_path());
  if (pcl::io::savePCDFileBinary(opt.output, *denoised) != 0) {
    std::cerr << "Failed to write " << opt.output << "\n";
    return 1;
  }

  std::cout << "Saved " << opt.output << " points=" << denoised->size()
            << " raw=" << merged->size() << "\n";
  return 0;
}
