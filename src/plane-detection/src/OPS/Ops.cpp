/**
 * Implementation of the following paper:
 * Oriented Point Sampling for OPSPlane Detection in Unorganized Point Clouds, Bo Sun and Philippos Mordohai (2019)
 *
 * by Victor AMBLARD
 */

#include "Ops.h"
#include "../RSPD/pointcloudio.hpp"

#define VERTICAL 1
#define HORIZONTAL 2
#define OTHER 3

const std::string orientationStr[4] = {"", "VERTICAL", "HORIZONTAL", "OTHER"};

double CLIP_ANGLE(double angle)
{
    if (angle > M_PI)
        angle -= M_PI;
    if (angle < M_PI && angle > M_PI / 2)
        angle = M_PI - angle;

    return angle;
}

std::vector<int> getNearestNeighbors(const int idx,
                                     const pcl::PointCloudXYZ::Ptr cloud,
                                     const int K,
                                     const pcl::KdTreeFLANN<pcl::PointXYZ> &kdTree)
{
    pcl::PointXYZ searchPoint = cloud->points[idx];
    std::vector<int> idxNeighbors(K);
    std::vector<float> pointsNNSquaredDist(K);
    kdTree.nearestKSearch(searchPoint, K, idxNeighbors, pointsNNSquaredDist);

    return idxNeighbors;
}

float getDistanceToOPSPlane(const Eigen::Vector3f &p,
                            const OPSPlane &P)
{
    Eigen::Vector3f diff = (p - P.second.second);
    float distance = std::fabs(diff.dot(P.second.first)); // normal vector is already of norm 1

    return distance;
}

float getDistanceToOPSPlane(const int id,
                            const int oId,
                            const pcl::PointCloudXYZ::Ptr cloud,
                            const Eigen::Vector3f &normal)
{
    Eigen::Vector3f diff = (cloud->points[id].getVector3fMap() - cloud->points[oId].getVector3fMap());
    float distance = std::fabs(diff.dot(normal)); // normal vector is already of norm 1

    return distance;
}
Eigen::Vector3f computeGlobalSVD(const std::vector<pcl::PointXYZ> &allPoints,
                                 const Eigen::Vector3f &centroid)
{
    int N = allPoints.size();
    Eigen::MatrixXd A(3, N);

    for (int i = 0; i < N; ++i)
    {
        Eigen::Vector3d eigPoint = allPoints.at(i).getVector3fMap().cast<double>() - centroid.cast<double>();
        A.block(0, i, 3, 1) = eigPoint;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV | Eigen::ComputeFullU);
    svd.computeU();
    Eigen::Vector3d normal(svd.matrixU()(2, 0), svd.matrixU()(2, 1), svd.matrixU()(2, 2));
    return normal.cast<float>().normalized();
}
Eigen::Vector3f computeSVDNormal(const std::vector<int> &nnIdx,
                                 const int piIdx,
                                 const pcl::PointCloudXYZ::Ptr cloud,
                                 const double sigma)
{
    Eigen::Vector3f pI = cloud->points[piIdx].getVector3fMap();
    Eigen::Matrix3f Mi = Eigen::Matrix3f::Zero();

    for (auto it = nnIdx.begin(); it != nnIdx.end(); ++it)
    {
        if (*it != piIdx) // All neighbors that are not the reference point
        {
            Eigen::Vector3f qij = cloud->points[*it].getVector3fMap();
            double sqNorm = (qij - pI).squaredNorm();
            double weight = std::exp(-sqNorm / (2 * pow(sigma, 2))); // weight factor
            Eigen::Matrix3f curMat = weight * 1 / sqNorm * (qij - pI) * (qij - pI).transpose();
            Mi += curMat;
        }
    }

    Eigen::Vector3f normal;
    Eigen::EigenSolver<Eigen::Matrix3f> es;
    es.compute(Mi, true);

    auto eVals = es.eigenvalues();
    float minEig = std::numeric_limits<float>::max();
    auto eVec = es.eigenvectors();

    for (unsigned int i = 0; i < eVals.size(); ++i)
    {
        if (eVals(i).real() < minEig) // eigenvalues are real
        {
            minEig = eVals(i).real();
            auto complexNormal = eVec.block(0, i, 3, 1);
            normal = Eigen::Vector3f(complexNormal(0, 0).real(), complexNormal(1, 0).real(), complexNormal(2, 0).real());
        }
    }

    return normal;
}
// Implements Algorithm 1 from the paper
std::pair<int, std::set<int>> detectCloud(const pcl::PointCloudXYZ::Ptr cloud,
                                          const std::vector<int> &samples,
                                          const std::vector<Eigen::Vector3f> &allNormals,
                                          const std::vector<int> &allOrientations,
                                          const bool ground,
                                          const double threshDistOPSPlane,
                                          const int threshInliers,
                                          const float threshAngle,
                                          const float p,
                                          const int defaultOrientation)
{
    std::chrono::time_point<std::chrono::system_clock> tStart = std::chrono::system_clock::now();

    int Ncloud = cloud->points.size();
    int Ns = samples.size();

    int iIter = 0;
    int nIter = Ncloud;
    int curMaxInliers = 0;
    std::set<int> bestInliers = {};
    int idxOI = -1;
    bool converged = true;
    int maxIter = 3000;
    size_t nInliers = 0;

    while (iIter < nIter)
    {
        int randIdx = std::rand() % Ns;
        idxOI = samples.at(randIdx);
        std::set<int> inliers;

        if (allOrientations.at(randIdx) == defaultOrientation)
        {

            for (int iPoint = 0; iPoint < Ns; ++iPoint)
            {
                if (iPoint != randIdx && CLIP_ANGLE(std::acos(allNormals.at(iPoint).dot(allNormals.at(randIdx)))) < threshAngle)
                {
                    double dist = getDistanceToOPSPlane(idxOI, samples.at(iPoint), cloud, allNormals.at(randIdx));
                    if (dist < threshDistOPSPlane)
                    {
                        inliers.insert(iPoint); // 2 criteria : distance to OPSPlane and common orientation
                    }
                }
            }

            nInliers = inliers.size();

            if (nInliers > threshInliers)
            {
                if (nInliers > curMaxInliers)
                {
                    curMaxInliers = nInliers;
                    bestInliers = inliers;
                    double e = 1 - (float)(nInliers) / Ns;
                    nIter = std::log(1 - p) / std::log(1 - (1 - e));
                }
            }
        }

        if (iIter > maxIter)
        {
            if (curMaxInliers == 0) // To avoid waiting 1h until the end of the loop
                converged = false;
            break;
        }
        ++iIter;
    }
    std::chrono::time_point<std::chrono::system_clock> tEnd = std::chrono::system_clock::now();
    double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count();
    if (converged)
        std::cerr << "Converged after " << iIter << " iterations to a OPSPlane with " << curMaxInliers << " inliers [process took: " << elapsedTime << " ms]" << std::endl;
    else
        std::cerr << "Failed to converge!" << std::endl;

    return std::make_pair(idxOI, bestInliers);
}

// void visualizeOPSPlanes(const std::vector<OPSPlane> &OPSPlanes,
//                         const pcl::PointCloudXYZ::Ptr remainingCloud)
// {
//     pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
//     viewer->setBackgroundColor(0, 0, 0);

//     pcl::PointCloudXYZRGB::Ptr coloredCloud(new pcl::PointCloudXYZRGB);
//     int i = 0;
//     for (auto OPSPlane : OPSPlanes)
//     {
//         for (pcl::PointXYZ point : OPSPlane.first)
//         {
//             pcl::PointXYZRGB p;
//             p.x = point.x;
//             p.y = point.y;
//             p.y = point.z;
//             p.r = r[(i * 61) % 255] * 255;
//             p.g = g[(i * 61) % 255] * 255;
//             p.b = b[(i * 61) % 255] * 255;

//             coloredCloud->points.push_back(p);
//             // coloredCloud->points.push_back(pcl::PointXYZRGB(point.x, point.y, point.z, r[(i * 61) % 255] * 255, g[(i * 61) % 255] * 255, b[(i * 61) % 255] * 255));
//             // coloredCloud->points.push_back(pcl::PointXYZRGB(point.x, point.y, point.z));
//         }
//         ++i;
//     }
//     std::cerr << "[INFO] Total " << coloredCloud->points.size() << " points in colored cloud" << std::endl;

//     viewer->addPointCloud<pcl::PointXYZRGB>(coloredCloud, "sample cloud");

//     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
//     viewer->addCoordinateSystem(1.0);
//     viewer->initCameraParameters();

//     while (!viewer->wasStopped())
//     {
//         viewer->spinOnce(100);
//     };
// }

std::pair<std::vector<Eigen::Vector3f>, std::vector<int>> computeAllNormals(const std::vector<int> &samples,
                                                                            const int K,
                                                                            const pcl::PointCloudXYZ::Ptr cloud,
                                                                            const float threshAngle)
{

    // Kd-tree construction
    pcl::KdTreeFLANN<pcl::PointXYZ> kdTree;
    kdTree.setInputCloud(cloud);

    std::vector<Eigen::Vector3f> allNormals;
    std::vector<int> allOrientations;
    allNormals.reserve(samples.size());
    allOrientations.reserve(samples.size());

    for (int idSampled : samples)
    {
        std::vector<int> idNN = getNearestNeighbors(idSampled, cloud, K, kdTree);
        Eigen::Vector3f curNormal = computeSVDNormal(idNN, idSampled, cloud);

        double angleToVerticalAxis = CLIP_ANGLE(std::acos(curNormal.dot(defaultParams::groundOrientation)));
        int orientationLabel;

        if (angleToVerticalAxis < threshAngle)
        {
            orientationLabel = VERTICAL;
        }
        else
        {
            if (angleToVerticalAxis > M_PI / 2 - threshAngle && angleToVerticalAxis < M_PI / 2)
            {
                orientationLabel = HORIZONTAL;
            }
            else
            {
                orientationLabel = OTHER;
            }
        }

        allNormals.push_back(curNormal);
        allOrientations.push_back(orientationLabel);
    }

    return std::make_pair(allNormals, allOrientations);
}

std::vector<OPSPlane> process(const pcl::PointCloudXYZ::Ptr cloud,
                              const bool verbose,
                              const double alphaS,
                              const int K,
                              const double threshDistOPSPlane,
                              const int threshInliers,
                              const float threshAngle,
                              const float p)
{

    if (verbose)
    {
        std::cerr << "======= Current parameters =======" << std::endl;
        std::cerr << "Sampling rate: " << alphaS * 100 << "%" << std::endl;
        std::cerr << "# nearest neighbors: " << K << std::endl;
        std::cerr << "Minimum number of inliers to approve a OPSPlane: " << threshInliers << std::endl;
        std::cerr << "Tolerance to vertical axis (ground detection) in degrees: " << threshAngle * 180 / M_PI << std::endl;
        std::cerr << "Distance threshold to OPSPlane: " << threshDistOPSPlane << std::endl;
        std::cerr << "RANSAC probability for adaptive number of iterations: " << p << std::endl;
        std::cerr << "==================================" << std::endl;
    }

    size_t sMini = 100;
    size_t sInliers = 0;

    std::vector<OPSPlane> OPSPlanes;
    std::pair<int, std::set<int>> inliersOPSPlane;
    pcl::PointCloudXYZ originalCloud(*cloud);
    int iIter = 0;

    // Step 1: Draw samples
    int Ncloud = cloud->points.size();
    int Ns = (int)round(alphaS * Ncloud);
    std::vector<int> ps;
    std::vector<int> allPointsIdx(Ncloud);
    for (int i = 0; i < Ncloud; ++i)
        allPointsIdx.at(i) = i;

    std::sample(allPointsIdx.begin(), allPointsIdx.end(), std::back_inserter(ps), Ns, std::mt19937{std::random_device{}()});

    // Step 2: Compute normals
    std::pair<std::vector<Eigen::Vector3f>, std::vector<int>> result = computeAllNormals(ps, K, cloud);
    std::vector<Eigen::Vector3f> allNormals = result.first;
    std::vector<int> allOrientations = result.second;

    // Step 3: Detect OPSPlanes

    int defaultOrientation;
    bool stop[3] = {false, false, false};
    do
    {

        if (iIter % 3 == 0)
        {
            if (stop[0])
                continue;
            defaultOrientation = HORIZONTAL;
        }
        if (iIter % 3 == 1)
        {
            if (stop[1])
                continue;
            defaultOrientation = VERTICAL;
        }
        if (iIter % 3 == 2)
        {
            if (stop[2])
                continue;
            defaultOrientation = OTHER;
        }
        inliersOPSPlane = detectCloud(cloud, ps, allNormals, allOrientations, false, threshDistOPSPlane, threshInliers, threshAngle, p, defaultOrientation);

        // Remove inliers from cloud
        pcl::ExtractIndices<pcl::PointXYZ> extractIndices;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

        sInliers = inliersOPSPlane.second.size();

        if (!sInliers)
            break; // Failed to converge

        Eigen::Vector3f centroid(0, 0, 0);
        std::vector<pcl::PointXYZ> curOPSPlane;

        for (auto it = inliersOPSPlane.second.begin(); it != inliersOPSPlane.second.end(); ++it)
        {
            centroid += (cloud->points[ps.at(*it)]).getVector3fMap() / sInliers;
            curOPSPlane.push_back(cloud->points[ps.at(*it)]);
        }

        std::vector<int> tmpPs;
        std::vector<Eigen::Vector3f> tmpNormals;
        std::vector<int> tmpOrientation;
        tmpOrientation.reserve(Ns - sInliers);
        tmpNormals.reserve(Ns - sInliers);
        tmpPs.reserve(Ns - sInliers);

        for (int i = 0; i < Ns; ++i)
        {
            if (inliersOPSPlane.second.find(i) == inliersOPSPlane.second.end()) // current point not in detected OPSPlane
            {
                tmpPs.push_back(ps.at(i));
                tmpNormals.push_back(allNormals.at(i));
                tmpOrientation.push_back(allOrientations.at(i));
            }
        }
        allOrientations = tmpOrientation;
        allNormals = tmpNormals;
        ps = tmpPs;

        Ns = ps.size();
        Eigen::Vector3f OPSPlaneNormal = computeGlobalSVD(curOPSPlane, centroid);
        OPSPlanes.push_back(std::make_pair(curOPSPlane, std::make_pair(OPSPlaneNormal, centroid)));

        if (verbose)
        {
            std::cerr << "[INFO] OPSPlane normal: (" << OPSPlaneNormal(0) << " ;" << OPSPlaneNormal(1) << " ; " << OPSPlaneNormal(2) << ")" << std::endl;
            std::cerr << "       Centroid: (" << centroid(0) << ";" << centroid(1) << ";" << centroid(2) << ")" << std::endl;
            std::cerr << "       Orientation: " << orientationStr[defaultOrientation] << std::endl;
        }

        if (ps.size() < threshInliers)
            stop[iIter % 3] = true;

        ++iIter;

    } while (!stop[0] || !stop[1] || !stop[2]);

    // Step 4: Merge OPSPlanes
    size_t OPSPlanesNumber;

    do
    {
        std::cerr << OPSPlanes.size() << std::endl;
        OPSPlanesNumber = OPSPlanes.size();
        mergeOPSPlanes(OPSPlanes);

    } while (OPSPlanes.size() != OPSPlanesNumber);
    PointCloudIO pointCloudIO;
    for (OPSPlane plane : OPSPlanes)
    {
        auto f = plane.first;
    }

    // Step 5: Visualize result
    // visualizeOPSPlanes(OPSPlanes, cloud);
}

void mergeOPSPlanes(std::vector<OPSPlane> &OPSPlanes)
{
    bool out = false;

    for (auto itA = OPSPlanes.begin(); itA != OPSPlanes.end();)
    {
        for (auto itB = itA; itB != OPSPlanes.end();)
        {
            if (itA != itB)
            {
                bool toMerge = compareOPSPlanes(*itA, *itB);

                if (toMerge)
                {
                    std::vector<pcl::PointXYZ> inliersA = (*itA).first;
                    std::vector<pcl::PointXYZ> inliersB = (*itB).first;
                    size_t nA = inliersA.size();
                    size_t nB = inliersB.size();

                    for (auto elem : inliersA)
                        itB->first.push_back(elem);

                    Eigen::Vector3f nCentroid = 1 / (nA + nB) * (nA * itA->second.second + nB * itB->second.second); // updated centroid
                    Eigen::Vector3f nNormal = computeGlobalSVD(itB->first, nCentroid);                               // Recompute normals with all inliers
                    itB->second.first = nNormal;
                    itB->second.second = nCentroid;
                    out = true;
                    itA = OPSPlanes.erase(itA);

                    break;
                }
                else
                {
                    ++itB;
                }
            }
            else
            {
                ++itB;
            }
        }
        if (!out)
            ++itA;
        else
            out = false;
    }
}

bool compareOPSPlanes(const OPSPlane &A,
                      const OPSPlane &B)
{
    float distA = getDistanceToOPSPlane(A.second.second, B);
    float distB = getDistanceToOPSPlane(B.second.second, A);
    float angle = CLIP_ANGLE(std::acos(A.second.first.dot(B.second.first)));

    return (distA < defaultParams::threshDistToOPSPlane) && (distB < defaultParams::threshDistToOPSPlane) && (angle < defaultParams::threshAngleToAxis);
}
