#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>
#include <functional>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::random_device rd;
std::mt19937 gen(rd());

double de_jong(const std::vector<double>& x) {
    return std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
}

double schwefel(const std::vector<double>& x) {
    double sum = 0.0;
    for (const auto& xi : x) {
        sum += -xi * std::sin(std::sqrt(std::abs(xi)));
    }
    return sum;
}

double rastrigin(const std::vector<double>& x) {
    double sum = 10 * x.size();
    for (const auto& xi : x) {
        sum += xi * xi - 10 * std::cos(2 * M_PI * xi);
    }
    return sum;
}

double michalewicz_no_param(const std::vector<double>& x) {
    int m = 10;
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += std::sin(x[i]) * std::pow(std::sin(((i + 1) * x[i] * x[i]) / M_PI), 2 * m);
    }
    return -sum;
}

std::vector<double> get_neighbor(const std::vector<double>& x, double step_size) {
    std::uniform_real_distribution<> dist(-step_size, step_size);
    std::vector<double> neighbor = x;
    for (auto& xi : neighbor) {
        xi += dist(gen);
    }
    return neighbor;
}

double hill_climbing_best(std::function<double(const std::vector<double>&)> func, int dim, double bounds_low, double bounds_high, int max_iter = 1000, double step_size = 0.1) {
    std::uniform_real_distribution<> dist(bounds_low, bounds_high);
    std::vector<double> x(dim);
    for (auto& xi : x) xi = dist(gen);

    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<double> best_neighbor = x;
        double best_value = func(x);
        for (int i = 0; i < 10; ++i) {
            auto neighbor = get_neighbor(x, step_size);
            double neighbor_value = func(neighbor);
            if (neighbor_value < best_value) {
                best_neighbor = neighbor;
                best_value = neighbor_value;
            }
        }
        if (best_value < func(x)) {
            x = best_neighbor;
        }
    }
    return func(x);
}

double hill_climbing_first(std::function<double(const std::vector<double>&)> func, int dim, double bounds_low, double bounds_high, int max_iter = 1000, double step_size = 0.1) {
    std::uniform_real_distribution<> dist(bounds_low, bounds_high);
    std::vector<double> x(dim);
    for (auto& xi : x) xi = dist(gen);

    for (int iter = 0; iter < max_iter; ++iter) {
        auto neighbor = get_neighbor(x, step_size);
        if (func(neighbor) < func(x)) {
            x = neighbor;
        }
    }
    return func(x);
}

double hill_climbing_worst(std::function<double(const std::vector<double>&)> func, int dim, double bounds_low, double bounds_high, int max_iter = 1000, double step_size = 0.1) {
    std::uniform_real_distribution<> dist(bounds_low, bounds_high);
    std::vector<double> x(dim);
    for (auto& xi : x) xi = dist(gen);

    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<double> worst_neighbor = x;
        double worst_value = func(x);
        for (int i = 0; i < 10; ++i) {
            auto neighbor = get_neighbor(x, step_size);
            double neighbor_value = func(neighbor);
            if (neighbor_value > worst_value) {
                worst_neighbor = neighbor;
                worst_value = neighbor_value;
            }
        }
        if (worst_value > func(x)) {
            x = worst_neighbor;
        }
    }
    return func(x);
}

double simulated_annealing(std::function<double(const std::vector<double>&)> func, int dim, double bounds_low, double bounds_high, int max_iter = 1000, double initial_temp = 1000, double cooling_rate = 0.99) {
    std::uniform_real_distribution<> dist(bounds_low, bounds_high);
    std::vector<double> x(dim);
    for (auto& xi : x) xi = dist(gen);

    double current_value = func(x);
    double temp = initial_temp;

    for (int iter = 0; iter < max_iter; ++iter) {
        auto neighbor = get_neighbor(x, 0.1);
        double neighbor_value = func(neighbor);

        if (neighbor_value < current_value || std::exp((current_value - neighbor_value) / temp) > dist(gen)) {
            x = neighbor;
            current_value = neighbor_value;
        }
        temp *= cooling_rate;
    }
    return func(x);
}

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double standard_deviation(const std::vector<double>& values, double mean_value) {
    double sum = 0.0;
    for (const auto& val : values) {
        sum += std::pow(val - mean_value, 2);
    }
    return std::sqrt(sum / values.size());
}

void run_and_display_stats(std::function<double(const std::vector<double>&)> func, const std::string& func_name, double bounds_low, double bounds_high, const std::vector<int>& dimensions) {
    for (int dim : dimensions) {
        std::vector<double> results_best, results_first, results_worst, results_sa;
        for (int i = 0; i < 30; ++i) {
            results_best.push_back(hill_climbing_best(func, dim, bounds_low, bounds_high));
            results_first.push_back(hill_climbing_first(func, dim, bounds_low, bounds_high));
            results_worst.push_back(hill_climbing_worst(func, dim, bounds_low, bounds_high));
            results_sa.push_back(simulated_annealing(func, dim, bounds_low, bounds_high));
        }
        double mean_best = mean(results_best);
        double mean_first = mean(results_first);
        double mean_worst = mean(results_worst);
        double mean_sa = mean(results_sa);

        double std_dev_best = standard_deviation(results_best, mean_best);
        double std_dev_first = standard_deviation(results_first, mean_first);
        double std_dev_worst = standard_deviation(results_worst, mean_worst);
        double std_dev_sa = standard_deviation(results_sa, mean_sa);

        std::cout << "Function: " << func_name << " - Dimension: " << dim << "\n";
        std::cout << "Algorithm      Min        Max        Mean       Std Dev\n";
        std::cout << "HC - Best      " << *std::min_element(results_best.begin(), results_best.end()) << "   "
            << *std::max_element(results_best.begin(), results_best.end()) << "   "
            << mean_best << "   " << std_dev_best << "\n";
        std::cout << "HC - First     " << *std::min_element(results_first.begin(), results_first.end()) << "   "
            << *std::max_element(results_first.begin(), results_first.end()) << "   "
            << mean_first << "   " << std_dev_first << "\n";
        std::cout << "HC - Worst     " << *std::min_element(results_worst.begin(), results_worst.end()) << "   "
            << *std::max_element(results_worst.begin(), results_worst.end()) << "   "
            << mean_worst << "   " << std_dev_worst << "\n";
        std::cout << "Sim. Annealing " << *std::min_element(results_sa.begin(), results_sa.end()) << "   "
            << *std::max_element(results_sa.begin(), results_sa.end()) << "   "
            << mean_sa << "   " << std_dev_sa << "\n\n";
    }
}

int main() {
    std::vector<int> dimensions = { 5, 10, 30 };

    run_and_display_stats(de_jong, "De Jong", -5.12, 5.12, dimensions);
    run_and_display_stats(schwefel, "Schwefel", -500, 500, dimensions);
    run_and_display_stats(rastrigin, "Rastrigin", -5.12, 5.12, dimensions);
    run_and_display_stats(michalewicz_no_param, "Michalewicz", 0, M_PI, dimensions);

    return 0;
}
