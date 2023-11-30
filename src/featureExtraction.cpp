#include "utility.h"
#include "lio_sam/cloud_info.h"

// Structure to store smoothness value and corresponding index
struct smoothness_t {
    float value;
    size_t ind;
};

// Comparator for sorting smoothness values
struct by_value {
    bool operator()(smoothness_t const&left, smoothness_t const&right) {
        return left.value < right.value;
    }
};

// FeatureExtraction class derived from ParamServer
class FeatureExtraction : public ParamServer {
public:
    // ROS subscribers and publishers
    ros::Subscriber subLaserCloudInfo;
    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;

    // Point clouds for extracted features and filtered points
    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    // Voxel grid filter for downsampling
    pcl::VoxelGrid<PointType> downSizeFilter;

    // Custom cloud info and standard ROS header
    lio_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    // Variables for feature extraction
    std::vector<smoothness_t> cloudSmoothness;
    float* cloudCurvature;
    int* cloudNeighborPicked;
    int* cloudLabel;

    FeatureExtraction() {
        // Initialize ROS subscribers and publishers
        subLaserCloudInfo = nh.subscribe<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1,
                                                              &FeatureExtraction::laserCloudInfoHandler, this,
                                                              ros::TransportHints().tcpNoDelay());

        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_corner", 1);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_surface", 1);

        initializationValue();
    }

    // Function to initialize variables
    void initializationValue() {
        cloudSmoothness.resize(N_SCAN * Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN * Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN * Horizon_SCAN]; // cloudNeighborPicked is an array used to mark points
        // that are either occluded or notsuitable for feature extraction due to their position or the nature of the
        // depth data around them. This is an essential step to ensure the reliability of the features extracted in
        // later stages.

        cloudLabel = new int[N_SCAN * Horizon_SCAN];
    }

    // Handler for incoming laser cloud info
    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr&msgIn) {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        publishFeatureCloud();
    }

    // Function to calculate the smoothness of each point in the cloud
    void calculateSmoothness() {
        int cloudSize = extractedCloud->points.size(); // Get the number of points in the cloud
        for (int i = 5; i < cloudSize - 5; i++) {
            // Loop over points, avoiding the first and last 5 points
            float diffRange = cloudInfo.pointRange[i - 5] + cloudInfo.pointRange[i - 4]
                              + cloudInfo.pointRange[i - 3] + cloudInfo.pointRange[i - 2]
                              + cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i] * 10
                              + cloudInfo.pointRange[i + 1] + cloudInfo.pointRange[i + 2]
                              + cloudInfo.pointRange[i + 3] + cloudInfo.pointRange[i + 4]
                              + cloudInfo.pointRange[i + 5];

            // Calculate the curvature as the square of diffRange
            // Curvature is used as a measure of smoothness
            cloudCurvature[i] = diffRange * diffRange; //diffX * diffX + diffY * diffY + diffZ * diffZ;

            // Initialize the neighbor picked and label arrays
            cloudNeighborPicked[i] = 0; // 0 indicates the point has not been picked yet
            cloudLabel[i] = 0; // 0 indicates the point has not been labeled yet

            // Store the smoothness value and the corresponding index for later sorting
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i]; // Smoothness value
            cloudSmoothness[i].ind = i; // Point index
        }
    }

    // Function to mark occluded points in the cloud
    void markOccludedPoints() {
        int cloudSize = extractedCloud->points.size(); // Total number of points in the cloud
        // mark occluded points and parallel beam points
        // Loop over the points, avoiding the first and last 5 points
        for (int i = 5; i < cloudSize - 6; ++i) {
            // occluded points
            // Depth values for the current and the next point
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i + 1];

            // Calculate the difference in column indices between adjacent points
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i + 1] - cloudInfo.pointColInd[i]));

            // Check if points are close enough in the image (less than 10 pixels apart)
            if (columnDiff < 10) {
                // 10 pixel diff in range image
                // If the depth difference between adjacent points is significant
                // it implies one point is occluding the other
                if (depth1 - depth2 > 0.3) {
                    // Mark the current and previous points as occluded
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }
                else if (depth2 - depth1 > 0.3) {
                    // Mark the current and next points as occluded
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            // Check for parallel beam points
            // These are points where there is a sudden change in depth, indicating
            // potential noise or points from edges or corners
            float diff1 = std::abs(float(cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i + 1] - cloudInfo.pointRange[i]));

            // If the change in depth is significant compared to the point's range,
            // mark it as picked (i.e., not to be used for feature extraction)
            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    // Function to extract features (edges and surfaces) from the cloud
    void extractFeatures() {
        cornerCloud->clear(); // Clear previous corner points
        surfaceCloud->clear(); // Clear previous surface points

        // Temporary point cloud to store surface points for each scan
        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < N_SCAN; i++) {
            surfaceCloudScan->clear();

            // Divide each scan into 6 segments for processing
            for (int j = 0; j < 6; j++) {
                // sp and ep are start and end points of the segment in the scan
                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                // Sort points in the segment by smoothness (curvature)
                std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, by_value());

                // Extract corner points
                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--) {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold) {
                        largestPickedNum++;
                        if (largestPickedNum <= 20) {
                            // Pick the first 20 points
                            cloudLabel[ind] = 1; // Label as corner point
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        }
                        else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1; // Mark as picked
                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(
                                int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }

                        // Avoid picking points that are too close to each other in the scan
                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(
                                int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // Extract surface points
                for (int k = sp; k <= ep; k++) {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold) {
                        cloudLabel[ind] = -1; // Label as surface point
                        cloudNeighborPicked[ind] = 1; // Mark as picked

                        // Avoid picking points that are too close to each other in the scan
                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(
                                int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(
                                int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // Collect all surface points
                for (int k = sp; k <= ep; k++) {
                    if (cloudLabel[k] <= 0) {
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            // Downsample the surface points to reduce computation
            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            // Combine the downsampled surface points from all
            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    // Helper function to free memory used by cloud info
    void freeCloudInfoMemory() {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    // Function to publish the extracted features
    void publishFeatureCloud() {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner = publishCloud(pubCornerPoints, cornerCloud, cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};


int main(int argc, char** argv) {
    ros::init(argc, argv, "lio_sam");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m");

    ros::spin();

    return 0;
}
