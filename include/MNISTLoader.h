#ifndef MNISTLOADER_H
#define MNISTLOADER_H

#include <string>
#include <vector>
#include <fstream> 
#include "Matrix.h" 

struct MNISTDataset {
    std::vector<Matrix> images;   
    std::vector<Matrix> labels;   
    int number_of_items;
    int image_rows;
    int image_cols;
};

class MNISTLoader {
public:
    static MNISTDataset load(const std::string& image_path, const std::string& label_path, int max_items = 0);

private:
    static int32_t read_int_big_endian(std::ifstream& ifs);
    static std::vector<Matrix> load_images(const std::string& path, int& number_of_images, int& image_rows, int& image_cols, int max_items);
    static std::vector<Matrix> load_labels(const std::string& path, int& number_of_labels, int max_items);
};

#endif 