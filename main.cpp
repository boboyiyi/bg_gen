#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include "./seamless_cloning.h"
#include "./string_utils.h"
#include "./aligner.h"
#include "./histogram_matching.h"

void
T_write(std::string T_file_name, Eigen::MatrixXf& T) {
    std::ofstream fo(T_file_name);
    for(int i = 0; i < T.rows(); i++) {
        for(int j = 0; j < T.cols(); j++) {
            fo << T(i,j) << " ";
        }
        fo << std::endl;
    }
    fo.close(); fo.clear();
}

int main(int argc, char **argv) {
    if (argc != 10) {
        std::cout << "Usage: bg_gen method ids img_dir pts_dir out_dir range_smooth" << std::endl;
        return 1;
    }
    std::string method = argv[1];
    if (method != "umeyama" && method != "ls") {
        std::cout << "Unknown aligner!" << std::endl;
        return 1;
    }
    int r_smooth = atoi(argv[9]);
    if (r_smooth < 0 || r_smooth > 5) {
        std::cout << "The range of smooth should be [0 - 5]" << std::endl;
        return 1;
    }
    std::vector<std::string> names;
    std::ifstream fi(argv[2]);
    std::string name = "";
    while (!fi.eof()) {
        std::getline(fi, name);
        name = tiny_trim(name);
        if (name.size() > 0) {
            names.push_back(name);
        }
    }

    fi.close(); fi.clear();
    aligner *_aligner = new aligner();
    std::string src_img_file_name = argv[3];
    std::string src_pts_file_name = argv[4];
    std::string mask_file_name = argv[5];
    std::string img_dir = argv[6];
    std::string pts_dir = argv[7];
    std::string out_dir = argv[8];
    std::string command = creat_dir;
    std::string cmd = "";
    cmd = command + out_dir;
    system(cmd.c_str());
    std::vector<Eigen::MatrixXf> Ts;
    
    for (int i = 0; i < names.size(); ++i) {
        std::string pts_file_name = pts_dir + delimiter + names[i] + ".pts";
        Eigen::MatrixXf T;
        if (method == "umeyama") {
            T = _aligner->umeyama(src_pts_file_name, pts_file_name);
        }
        else {
            T = _aligner->ls(src_pts_file_name, pts_file_name);
        }
        Ts.push_back(T);
    }
    std::vector<Eigen::MatrixXf> smooth_Ts;
    bool do_smooth = false;
    for (int i = 0; i < names.size(); ++i) {
        std::cout << i << std::endl;
        if (i - r_smooth >=0 && i + r_smooth < names.size()) {
            do_smooth = true;
        }
        else {
            do_smooth = false;
        }
        if (do_smooth && r_smooth > 0) {
            Eigen::MatrixXf tmp = Eigen::MatrixXf::Zero(2, 3);
            for (int k = -r_smooth; k <= r_smooth; ++k) {
                tmp += Ts[i + k];
            }
            tmp /= (r_smooth * 2 + 1);
            smooth_Ts.push_back(tmp);
        }
        else {
            smooth_Ts.push_back(Ts[i]);
        }
    }
    Ts.resize(0);
    cv::Mat _T(2, 3, CV_32FC1);
    cv::Mat src = cv::imread(src_img_file_name, 1);
    cv::Mat mask = cv::imread(mask_file_name, 0);
    Cloning obj;
    histogram_core hc;
    for (int i = 0; i < names.size(); ++i) {
        std::string img_file_name = img_dir + delimiter + names[i] + ".jpg";
        std::cout << names[i] << std::endl;
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 3; ++k) {
                _T.at<float>(j, k) = smooth_Ts[i](j, k);
            }
        }
        cv::Mat dst = cv::imread(img_file_name, 1);
        cv::Mat warped_src, warped_mask, out;
        warpAffine(src, warped_src, _T, cv::Size(src.cols, src.rows));
        warpAffine(mask, warped_mask, _T, cv::Size(src.cols, src.rows));
        cv::Mat non_zero_pts;
        cv::findNonZero(warped_mask, non_zero_pts);
        cv::Rect roi_rect = cv::boundingRect(non_zero_pts);
        roi_rect.x -= 10; roi_rect.y -= 10; roi_rect.width += 20; roi_rect.height += 20;
        // cv::rectangle(warped_mask, roi_rect, cv::Scalar::all(30));
        // cv::imwrite("wm.jpg", warped_mask);
        // cv::imwrite("wc.jpg", warped_src);
        out.create(dst.size(), CV_8UC3);
        dst.copyTo(out);

        // histogram match的方法能很明显看出边界
        // hc.histogram_matching(dst, warped_src, warped_mask, out);
        obj.normalClone(dst(roi_rect), warped_src(roi_rect), warped_mask(roi_rect), out(roi_rect), cv::NORMAL_CLONE);
        std::string out_img_file_name = out_dir + delimiter + names[i] + ".jpg";
        cv::imwrite(out_img_file_name, out);
    }
    delete _aligner;
    return 0;
}