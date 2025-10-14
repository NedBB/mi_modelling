#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <iostream>

// Include your existing MI modeling classes
#include "../include/FitzHughNagumo.h"
#include "../include/DTM.h"

using namespace emscripten;

// WebAssembly wrapper for FitzHugh-Nagumo model
class WASMFitzHughNagumo {
private:
    std::unique_ptr<FitzHughNagumo> model;
    
public:
    WASMFitzHughNagumo(int width, int height, double dt = 0.01) {
        model = std::make_unique<FitzHughNagumo>(width, height, dt);
    }
    
    void initialize() {
        model->initialize();
    }
    
    void setParameters(double a, double b, double c, double d) {
        model->setParameters(a, b, c, d);
    }
    
    void setDiffusionCoefficients(double du, double dv) {
        model->setDiffusionCoefficients(du, dv);
    }
    
    void addStimulus(int x, int y, double strength, double duration) {
        model->addStimulus(x, y, strength, duration);
    }
    
    void run(int steps) {
        model->run(steps);
    }
    
    void step() {
        model->step();
    }
    
    double getTime() const {
        return model->getTime();
    }
    
    // Return membrane potential data as JavaScript array
    emscripten::val getMembranePotential() const {
        const auto& data = model->getU();
        emscripten::val result = emscripten::val::array();
        
        for (const auto& row : data) {
            emscripten::val jsRow = emscripten::val::array();
            for (double value : row) {
                jsRow.call<void>("push", value);
            }
            result.call<void>("push", jsRow);
        }
        
        return result;
    }
    
    // Return recovery variable data as JavaScript array
    emscripten::val getRecoveryVariable() const {
        const auto& data = model->getV();
        emscripten::val result = emscripten::val::array();
        
        for (const auto& row : data) {
            emscripten::val jsRow = emscripten::val::array();
            for (double value : row) {
                jsRow.call<void>("push", value);
            }
            result.call<void>("push", jsRow);
        }
        
        return result;
    }
    
    // Get dimensions
    emscripten::val getDimensions() const {
        emscripten::val result = emscripten::val::object();
        result.set("width", 100);  // Default for demo
        result.set("height", 100); // Default for demo
        return result;
    }
    
    // Save state to file
    bool saveState(const std::string& filename) const {
        return model->saveState(filename);
    }
    
    // Load state from file
    bool loadState(const std::string& filename) {
        return model->loadState(filename);
    }
};

// WebAssembly wrapper for DTM solver
class WASMDTM {
private:
    std::unique_ptr<DTM> dtm;
    
public:
    WASMDTM(int width, int height, double spatial_step = 1.0) {
        dtm = std::make_unique<DTM>(width, height, spatial_step);
    }
    
    bool loadFromFile(const std::string& filename) {
        return dtm->loadFromFile(filename);
    }
    
    bool saveToFile(const std::string& filename) const {
        return dtm->saveToFile(filename);
    }
    
    double getElevation(int x, int y) const {
        return dtm->getElevation(x, y);
    }
    
    void setElevation(int x, int y, double elevation) {
        dtm->setElevation(x, y, elevation);
    }
    
    emscripten::val getDimensions() const {
        auto dims = dtm->getDimensions();
        emscripten::val result = emscripten::val::object();
        result.set("width", dims.first);
        result.set("height", dims.second);
        return result;
    }
    
    double getCellSize() const {
        return dtm->getCellSize();
    }
    
    double calculateSlope(int x, int y) const {
        return dtm->calculateSlope(x, y);
    }
    
    double calculateAspect(int x, int y) const {
        return dtm->calculateAspect(x, y);
    }
};

// File processing functions
class FileProcessor {
public:
    static emscripten::val processECGData(emscripten::val jsData) {
        // Convert JavaScript array to C++ vector
        std::vector<double> data;
        int length = jsData["length"].as<int>();
        
        for (int i = 0; i < length; i++) {
            data.push_back(jsData[i].as<double>());
        }
        
        // Process ECG data
        std::vector<double> processed;
        double mean = 0.0;
        double sum = 0.0;
        
        // Calculate mean
        for (double value : data) {
            sum += value;
        }
        mean = sum / data.size();
        
        // Apply simple high-pass filter (remove baseline)
        for (double value : data) {
            processed.push_back(value - mean);
        }
        
        // Return processed data as JavaScript array
        emscripten::val result = emscripten::val::array();
        for (double value : processed) {
            result.call<void>("push", value);
        }
        
        return result;
    }
    
    static emscripten::val detectRPeaks(emscripten::val jsData) {
        std::vector<double> data;
        int length = jsData["length"].as<int>();
        
        for (int i = 0; i < length; i++) {
            data.push_back(jsData[i].as<double>());
        }
        
        // Simple R-peak detection
        std::vector<int> peaks;
        double threshold = 0.0;
        double max_val = 0.0;
        
        // Find maximum value
        for (double value : data) {
            if (std::abs(value) > max_val) {
                max_val = std::abs(value);
            }
        }
        
        threshold = max_val * 0.7; // 70% of maximum
        
        // Detect peaks
        for (int i = 1; i < data.size() - 1; i++) {
            if (data[i] > threshold && 
                data[i] > data[i-1] && 
                data[i] > data[i+1]) {
                peaks.push_back(i);
            }
        }
        
        // Return peaks as JavaScript array
        emscripten::val result = emscripten::val::array();
        for (int peak : peaks) {
            result.call<void>("push", peak);
        }
        
        return result;
    }
    
    static emscripten::val calculateECGMetrics(emscripten::val jsData) {
        std::vector<double> data;
        int length = jsData["length"].as<int>();
        
        for (int i = 0; i < length; i++) {
            data.push_back(jsData[i].as<double>());
        }
        
        // Calculate metrics
        double min_val = *std::min_element(data.begin(), data.end());
        double max_val = *std::max_element(data.begin(), data.end());
        
        double sum = 0.0;
        for (double value : data) {
            sum += value;
        }
        double mean = sum / data.size();
        
        double variance = 0.0;
        for (double value : data) {
            variance += (value - mean) * (value - mean);
        }
        double std_dev = std::sqrt(variance / data.size());
        
        // Return metrics as JavaScript object
        emscripten::val result = emscripten::val::object();
        result.set("min", min_val);
        result.set("max", max_val);
        result.set("mean", mean);
        result.set("std", std_dev);
        result.set("range", max_val - min_val);
        
        return result;
    }
};

// Global simulation runner functions
extern "C" {
    EMSCRIPTEN_KEEPALIVE
    double run_fitzhugh_nagumo_simulation(int width, int height, int steps, double dt) {
        FitzHughNagumo model(width, height, dt);
        model.initialize();
        model.setParameters(0.1, 0.5, 1.0, 0.0);
        model.setDiffusionCoefficients(0.1, 0.0);
        
        auto start = std::chrono::high_resolution_clock::now();
        model.run(steps);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return duration.count();
    }
    
    EMSCRIPTEN_KEEPALIVE
    double run_dtm_simulation(int width, int height, double spatial_step) {
        DTM dtm(width, height, spatial_step);
        
        // Fill with sample data
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double center_x = width / 2.0;
                double center_y = height / 2.0;
                double distance = std::sqrt((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y));
                double elevation = 100.0 * std::exp(-distance / 10.0);
                dtm.setElevation(x, y, elevation);
            }
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Calculate terrain properties
        double total_slope = 0.0;
        int count = 0;
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                total_slope += dtm.calculateSlope(x, y);
                count++;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        return duration.count();
    }
}

// Emscripten bindings to expose classes to JavaScript
EMSCRIPTEN_BINDINGS(mi_modeling_wasm) {
    // FitzHugh-Nagumo bindings
    class_<WASMFitzHughNagumo>("FitzHughNagumo")
        .constructor<int, int, double>()
        .function("initialize", &WASMFitzHughNagumo::initialize)
        .function("setParameters", &WASMFitzHughNagumo::setParameters)
        .function("setDiffusionCoefficients", &WASMFitzHughNagumo::setDiffusionCoefficients)
        .function("addStimulus", &WASMFitzHughNagumo::addStimulus)
        .function("run", &WASMFitzHughNagumo::run)
        .function("step", &WASMFitzHughNagumo::step)
        .function("getTime", &WASMFitzHughNagumo::getTime)
        .function("getMembranePotential", &WASMFitzHughNagumo::getMembranePotential)
        .function("getRecoveryVariable", &WASMFitzHughNagumo::getRecoveryVariable)
        .function("getDimensions", &WASMFitzHughNagumo::getDimensions)
        .function("saveState", &WASMFitzHughNagumo::saveState)
        .function("loadState", &WASMFitzHughNagumo::loadState);
    
    // DTM bindings
    class_<WASMDTM>("DTM")
        .constructor<int, int, double>()
        .function("loadFromFile", &WASMDTM::loadFromFile)
        .function("saveToFile", &WASMDTM::saveToFile)
        .function("getElevation", &WASMDTM::getElevation)
        .function("setElevation", &WASMDTM::setElevation)
        .function("getDimensions", &WASMDTM::getDimensions)
        .function("getCellSize", &WASMDTM::getCellSize)
        .function("calculateSlope", &WASMDTM::calculateSlope)
        .function("calculateAspect", &WASMDTM::calculateAspect);
    
    // File processor bindings
    class_<FileProcessor>("FileProcessor")
        .class_function("processECGData", &FileProcessor::processECGData)
        .class_function("detectRPeaks", &FileProcessor::detectRPeaks)
        .class_function("calculateECGMetrics", &FileProcessor::calculateECGMetrics);
}

