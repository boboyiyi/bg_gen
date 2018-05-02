#include "./aligner.h"

aligner::aligner() {
}

aligner::~aligner() {
}

void
aligner::pts_read(std::string &pts_file_name, Eigen::MatrixXf &point) {
    std::ifstream fi(pts_file_name);
    std::string line;
    std::getline(fi, line); std::getline(fi, line); std::getline(fi, line);
    for (int i = 0; i < NUM_PTS; ++i) {
        fi >> point(i, 0) >> point(i, 1);
    }
    fi.close();
    fi.clear();
}

void
aligner::pts_write(std::string &pts_name, std::vector<cv::Point2f> &points) {
    std::ofstream fo(pts_name);
    fo << "version: 1\nn_points:" << NUM_PTS << "\n{\n";
    for (int i = 0; i < NUM_PTS; ++i) {
        fo << points[i].x << " " << points[i].y << std::endl;
    }
    fo << "}" << std::endl;
    fo.close(); fo.clear();
}

Eigen::MatrixXf
aligner::umeyama(std::string &src_pts_file_name, std::string &dst_pts_file_name) {
    Eigen::MatrixXf src_pts = Eigen::MatrixXf::Zero(NUM_PTS, 2);
    Eigen::MatrixXf dst_pts = Eigen::MatrixXf::Zero(NUM_PTS, 2);
    pts_read(src_pts_file_name, src_pts);
    pts_read(dst_pts_file_name, dst_pts);
    Eigen::MatrixXf T = Eigen::umeyama(src_pts.transpose(), dst_pts.transpose(), true);
    return T.block<2,3>(0,0);
}

Eigen::MatrixXf
aligner::ls(std::string &src_pts_file_name, std::string &dst_pts_file_name) {
    Eigen::MatrixXf src_pts = Eigen::MatrixXf::Zero(NUM_PTS, 2);
    Eigen::MatrixXf dst_pts = Eigen::MatrixXf::Zero(NUM_PTS, 2);

    pts_read(src_pts_file_name, src_pts);
    pts_read(dst_pts_file_name, dst_pts);

    // open below two lines in order to use pts without the region of jaw
    // src_pts = src_pts.block<NUM_ALIGN_PTS, 2>(NUM_PTS - NUM_ALIGN_PTS, 0).eval();
    // dst_pts = dst_pts.block<NUM_ALIGN_PTS, 2>(NUM_PTS - NUM_ALIGN_PTS, 0).eval();

    Eigen::MatrixXf T = Eigen::MatrixXf::Zero(2, 3);

    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(2 * src_pts.rows(), 6);
    Eigen::MatrixXf b(2 * dst_pts.rows(), 1);
    for (int i = 0; i < src_pts.rows(); ++i) {
        A.block<1, 2>(2 * i, 0) = src_pts.row(i);
        A.block<1, 2>((2 * i) + 1, 3) = src_pts.row(i);
        A(2 * i, 2) = 1;
        A(2 * i + 1, 5) = 1;
    }
    for (int i = 0; i < dst_pts.rows(); ++i) {
        b(2 * i, 0) = dst_pts(i, 0);
        b(2 * i + 1, 0) = dst_pts(i, 1);
    }
    const Eigen::Matrix<float, 6, 1> k = A.colPivHouseholderQr().solve(b);
    T.block<1, 3>(0, 0) = k.segment<3>(0);
    T.block<1, 3>(1, 0) = k.segment<3>(3);
    std::cout << T << std::endl;
    return T;
}
