#include "./histogram_matching.h"

histogram_core::histogram_core() {
}

histogram_core::~histogram_core() {
}

void
histogram_core::histogram_matching(const cv::Mat &source_image, const cv::Mat &target_image, const cv::Mat &mask, cv::Mat &cloned) {
    // ref: https://github.com/hrastnik/FaceSwap
    // 这里source_image是指背景图像，而target_image指需要贴的部分，所以叫target，根据source_image的hist调整target的区域颜色
    std::memset(source_hist_int, 0, sizeof(int) * 3 * 256);
    std::memset(target_hist_int, 0, sizeof(int) * 3 * 256);
    for (size_t i = 0; i < mask.rows; i++)
    {
        auto current_mask_pixel = mask.row(i).data;
        auto current_source_pixel = source_image.row(i).data;
        auto current_target_pixel = target_image.row(i).data;

        for (size_t j = 0; j < mask.cols; j++)
        {
            if (*current_mask_pixel != 0) {
                source_hist_int[0][*current_source_pixel]++;
                source_hist_int[1][*(current_source_pixel + 1)]++;
                source_hist_int[2][*(current_source_pixel + 2)]++;

                target_hist_int[0][*current_target_pixel]++;
                target_hist_int[1][*(current_target_pixel + 1)]++;
                target_hist_int[2][*(current_target_pixel + 2)]++;
            }

            // Advance to next pixel
            current_source_pixel += 3; 
            current_target_pixel += 3; 
            current_mask_pixel++; 
        }
    }

    // Calc CDF
    for (size_t i = 1; i < 256; i++)
    {
        source_hist_int[0][i] += source_hist_int[0][i - 1];
        source_hist_int[1][i] += source_hist_int[1][i - 1];
        source_hist_int[2][i] += source_hist_int[2][i - 1];

        target_hist_int[0][i] += target_hist_int[0][i - 1];
        target_hist_int[1][i] += target_hist_int[1][i - 1];
        target_hist_int[2][i] += target_hist_int[2][i - 1];
    }

    // Normalize CDF
    for (size_t i = 0; i < 256; i++)
    {
        source_histogram[0][i] = (source_hist_int[0][255] ? (float)source_hist_int[0][i] / source_hist_int[0][255] : 0);
        source_histogram[1][i] = (source_hist_int[1][255] ? (float)source_hist_int[1][i] / source_hist_int[1][255] : 0);
        source_histogram[2][i] = (source_hist_int[2][255] ? (float)source_hist_int[2][i] / source_hist_int[2][255] : 0);

        target_histogram[0][i] = (target_hist_int[0][255] ? (float)target_hist_int[0][i] / target_hist_int[0][255] : 0);
        target_histogram[1][i] = (target_hist_int[1][255] ? (float)target_hist_int[1][i] / target_hist_int[1][255] : 0);
        target_histogram[2][i] = (target_hist_int[2][255] ? (float)target_hist_int[2][i] / target_hist_int[2][255] : 0);
    }

    // Create lookup table
    
    auto binary_search = [&](const float needle, const float haystack[]) -> uint8_t
    {
        uint8_t l = 0, r = 255, m;
        while (l < r)
        {
            m = (l + r) / 2;
            if (needle > haystack[m])
                l = m + 1;
            else
                r = m - 1;
        }
        // TODO check closest value
        return m;
    };

    for (size_t i = 0; i < 256; i++)
    {
        LUT[0][i] = binary_search(target_histogram[0][i], source_histogram[0]);
        LUT[1][i] = binary_search(target_histogram[1][i], source_histogram[1]);
        LUT[2][i] = binary_search(target_histogram[2][i], source_histogram[2]);
    }

    source_image.copyTo(cloned);
    // cv::imwrite("before.jpg", cloned);
    // repaint pixels
    for (size_t i = 0; i < mask.rows; i++)
    {
        auto current_mask_pixel = mask.row(i).data;
        auto current_target_pixel = target_image.row(i).data;
        auto cloned_pixel = cloned.row(i).data;
        for (size_t j = 0; j < mask.cols; j++)
        {
            if (*current_mask_pixel != 0)
            {
                // std::cout << i << " " << j << std::endl;
                // std::cout << int(*current_target_pixel) << " " << int(LUT[0][*current_target_pixel]);
                *cloned_pixel = LUT[0][*current_target_pixel];
                *(cloned_pixel + 1) = LUT[1][*(current_target_pixel + 1)];
                *(cloned_pixel + 2) = LUT[2][*(current_target_pixel + 2)];
            }

            // Advance to next pixel
            current_target_pixel += 3;
            cloned_pixel += 3;
            current_mask_pixel++;
        }
    }
    // cv::imwrite("after.jpg", cloned);
}