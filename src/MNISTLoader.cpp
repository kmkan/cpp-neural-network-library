#include "MNISTLoader.h"
#include <stdexcept> 
#include <iostream>  
#include <vector>    

int32_t MNISTLoader::read_int_big_endian(std::ifstream& ifs) {
    int32_t val = 0;
    unsigned char bytes[4];
    if (!ifs.read(reinterpret_cast<char*>(bytes), 4)) {
        throw std::runtime_error("MNISTLoader: Failed to read 4 bytes for integer.");
    }
    val = (static_cast<int32_t>(bytes[0]) << 24) |
          (static_cast<int32_t>(bytes[1]) << 16) |
          (static_cast<int32_t>(bytes[2]) << 8)  |
          (static_cast<int32_t>(bytes[3]));
    return val;
}

std::vector<Matrix> MNISTLoader::load_images(const std::string& path, int& number_of_images, int& image_rows, int& image_cols, int max_items_to_load) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("MNISTLoader: Cannot open image file: " + path);
    }

    int32_t magic_number = read_int_big_endian(ifs);
    if (magic_number != 0x00000803) { 
        throw std::runtime_error("MNISTLoader: Invalid magic number in image file: " + path + ". Expected 2051, got " + std::to_string(magic_number));
    }

    number_of_images = read_int_big_endian(ifs);
    image_rows = read_int_big_endian(ifs);
    image_cols = read_int_big_endian(ifs);

    int items_to_read = number_of_images;
    if (max_items_to_load > 0 && max_items_to_load < number_of_images) {
        items_to_read = max_items_to_load;
        number_of_images = items_to_read; 
    }
    
    std::cout << "Loading " << items_to_read << " images (" 
              << image_rows << "x" << image_cols << ") from " << path << std::endl;

    std::vector<Matrix> images_data;
    images_data.reserve(items_to_read);

    int image_size = image_rows * image_cols;
    std::vector<unsigned char> buffer(image_size);

    for (int i = 0; i < items_to_read; ++i) {
        if (!ifs.read(reinterpret_cast<char*>(buffer.data()), image_size)) {
            throw std::runtime_error("MNISTLoader: Failed to read image data for image " + std::to_string(i) + " from " + path);
        }

        Matrix image_matrix(image_size, 1); 
        for (int j = 0; j < image_size; ++j) {
            image_matrix.setEntry(j, 0, static_cast<double>(buffer[j]) / 255.0);
        }
        images_data.push_back(image_matrix);

        if ((i + 1) % 10000 == 0 && items_to_read > 10000) {
            std::cout << "Loaded " << (i + 1) << "/" << items_to_read << " images..." << std::endl;
        }
    }
    ifs.close();
    return images_data;
}

std::vector<Matrix> MNISTLoader::load_labels(const std::string& path, int& number_of_labels, int max_items_to_load) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("MNISTLoader: Cannot open label file: " + path);
    }

    int32_t magic_number = read_int_big_endian(ifs);
    if (magic_number != 0x00000801) { 
        throw std::runtime_error("MNISTLoader: Invalid magic number in label file: " + path + ". Expected 2049, got " + std::to_string(magic_number));
    }

    number_of_labels = read_int_big_endian(ifs);
    
    int items_to_read = number_of_labels;
    if (max_items_to_load > 0 && max_items_to_load < number_of_labels) {
        items_to_read = max_items_to_load;
        number_of_labels = items_to_read; 
    }

    std::cout << "Loading " << items_to_read << " labels from " << path << std::endl;

    std::vector<Matrix> labels_data;
    labels_data.reserve(items_to_read);
    unsigned char label_buffer;

    for (int i = 0; i < items_to_read; ++i) {
        if (!ifs.read(reinterpret_cast<char*>(&label_buffer), 1)) {
            throw std::runtime_error("MNISTLoader: Failed to read label data for label " + std::to_string(i) + " from " + path);
        }

        Matrix label_matrix(10, 1, 0.0); 
        if (static_cast<int>(label_buffer) >= 0 && static_cast<int>(label_buffer) < 10) {
            label_matrix.setEntry(static_cast<int>(label_buffer), 0, 1.0); 
        } else {
            std::cerr << "Warning: Invalid label " << static_cast<int>(label_buffer) << " encountered at index " << i << std::endl;
        }
        labels_data.push_back(label_matrix);
         if ((i + 1) % 10000 == 0 && items_to_read > 10000) {
            std::cout << "Loaded " << (i + 1) << "/" << items_to_read << " labels..." << std::endl;
        }
    }
    ifs.close();
    return labels_data;
}

MNISTDataset MNISTLoader::load(const std::string& image_path, const std::string& label_path, int max_items) {
    MNISTDataset dataset;
    dataset.images = load_images(image_path, dataset.number_of_items, dataset.image_rows, dataset.image_cols, max_items);
    
    int num_labels_temp; 
    dataset.labels = load_labels(label_path, num_labels_temp, max_items);

    if (dataset.images.size() != dataset.labels.size()) {
        throw std::runtime_error("MNISTLoader: Mismatch between number of loaded images (" + std::to_string(dataset.images.size()) +
                                 ") and labels (" + std::to_string(dataset.labels.size()) + ").");
    }
    dataset.number_of_items = static_cast<int>(dataset.images.size());

    std::cout << "Successfully loaded " << dataset.number_of_items << " image-label pairs." << std::endl;
    if (!dataset.images.empty()) {
         std::cout << "Image dimensions: " << dataset.image_rows << "x" << dataset.image_cols 
                   << " (flattened to " << dataset.images[0].getRow() << "x" << dataset.images[0].getCol() << ")" << std::endl;
    }
    if (!dataset.labels.empty()) {
        std::cout << "Label dimensions (one-hot): " << dataset.labels[0].getRow() << "x" << dataset.labels[0].getCol() << std::endl;
    }

    return dataset;
}