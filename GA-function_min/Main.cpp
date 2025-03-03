#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <algorithm>
#include <iomanip> // Pentru formatare
#include <cmath>   // Pentru calcul deviație standard   
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Parametrii problemei
const int DIM = 30;            // Dimensiunea problemei (numărul de variabile)
const int POP_SIZE = 100;      // Mărimea populației
const int GENERATIONS = 10000;  // Numărul de generații
const double CROSSOVER_RATE = 0.8; // Probabilitatea de încrucișare
const double MUTATION_RATE = 0.1;  // Probabilitatea de mutație
const double DOMAIN_MIN = -5.12;  // Limita inferioară a domeniului (pentru Rastrigin și De Jong)
const double DOMAIN_MAX = 5.12;   // Limita superioară a domeniului (pentru Rastrigin și De Jong)

// Funcțiile obiectiv
double rastrigin(const vector<double>& x) {
    double result = 10.0 * x.size();
    for (double xi : x) {
        result += xi * xi - 10.0 * cos(2 * M_PI * xi);
    }
    return result;
}

double dejong(const vector<double>& x) {
    double result = 0.0;
    for (double xi : x) {
        result += xi * xi;
    }
    return result;
}

double schwefel(const vector<double>& x) {
    double result = 0;
    for (double xi : x) {
        result -= xi * sin(sqrt(abs(xi)));
    }
    return result;
}

double michalewicz(const vector<double>& x, int m = 10) {
    double result = 0.0;
    for (int i = 0; i < x.size(); i++) {
        result -= sin(x[i]) * pow(sin(((i + 1) * x[i] * x[i]) / M_PI), 2 * m);
    }
    return result;
}

// Funcția fitness (invers proporțională cu valoarea funcției obiectiv)
double fitness(const vector<double>& x, const string& function) {
    if (function == "rastrigin") {
        return 1.0 / (1.0 + rastrigin(x));
    }
    else if (function == "dejong") {
        return 1.0 / (1.0 + dejong(x));
    }
    else if (function == "schwefel") {
        return -schwefel(x);
        // return 1.0 / (1.0 + schwefel(x));
    }
    else if (function == "michalewicz") {
        return -michalewicz(x);
    }
    return 0.0; // Implicit, dacă funcția nu este recunoscută
}

vector<vector<double>> select_elite(const vector<vector<double>>& population, const vector<double>& fitness_values, int elite_size) {
    vector<pair<double, vector<double>>> fitness_population;
    for (size_t i = 0; i < population.size(); ++i) {
        fitness_population.emplace_back(fitness_values[i], population[i]);
    }

    // Sortează descrescător după fitness
    sort(fitness_population.begin(), fitness_population.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    // Extrage cei mai buni indivizi
    vector<vector<double>> elite;
    for (int i = 0; i < elite_size; ++i) {
        elite.push_back(fitness_population[i].second);
    }

    return elite;
}


// Inițializarea populației
vector<vector<double>> initialize_population(double domain_min, double domain_max) {
    vector<vector<double>> population(POP_SIZE, vector<double>(DIM));
    for (auto& individual : population) {
        for (double& gene : individual) {
            gene = domain_min + (domain_max - domain_min) * ((double)rand() / RAND_MAX);
        }
    }
    return population;
}

// Selecție prin ruletă
vector<double> roulette_selection(const vector<vector<double>>& population, const vector<double>& fitness_values) {

    double total_fitness = accumulate(fitness_values.begin(), fitness_values.end(), 0.0);
    double point = ((double)rand() / RAND_MAX) * total_fitness;
    double cumulative = 0.0;
    for (size_t i = 0; i < population.size(); i++) {
        cumulative += fitness_values[i];
        if (cumulative >= point) {
            return population[i];
        }
    }
    return population.back();
}

// Încrucișare uniformă
pair<vector<double>, vector<double>> crossover(const vector<double>& parent1, const vector<double>& parent2) {
    vector<double> offspring1 = parent1;
    vector<double> offspring2 = parent2;
    for (int i = 0; i < DIM; i++) { 
        if ((double)rand() / RAND_MAX < 0.5) {
            swap(offspring1[i], offspring2[i]);
        }
    }
    return { offspring1, offspring2 };
}

// Mutație
void mutate(vector<double>& individual, double domain_min, double domain_max) {
    for (double& gene : individual) {
        if ((double)rand() / RAND_MAX < MUTATION_RATE) {
            gene += (domain_max - domain_min) * ((double)rand() / RAND_MAX - 0.5);
            gene = min(max(gene, domain_min), domain_max); // Constrângere în domeniu
        }
    }
}

vector<double> genetic_algorithm(const string& function, double domain_min, double domain_max) {
    srand(time(0));
    vector<vector<double>> population = initialize_population(domain_min, domain_max);
    vector<double> best_individual;
    double best_fitness = -1e9;

    const int ELITE_SIZE = 10; // Numărul de indivizi păstrați prin elitism

    for (int generation = 0; generation < GENERATIONS; generation++) {
        vector<double> fitness_values(POP_SIZE);

        // Calculează fitness pentru toți indivizii
        for (int i = 0; i < POP_SIZE; i++) {
            fitness_values[i] = fitness(population[i], function);
            if (fitness_values[i] > best_fitness) {
                best_fitness = fitness_values[i];
                best_individual = population[i];
            }
        }

        // Selectează indivizii de elită
        vector<vector<double>> new_population = select_elite(population, fitness_values, ELITE_SIZE);

        // Creează restul populației prin încrucișare și mutație
        while (new_population.size() < POP_SIZE) {
            vector<double> parent1 = population[rand() % POP_SIZE]; // Selectăm aleatoriu
            vector<double> parent2 = population[rand() % POP_SIZE];

            if ((double)rand() / RAND_MAX < CROSSOVER_RATE) {
                auto [offspring1, offspring2] = crossover(parent1, parent2);

                if ((double)rand() / RAND_MAX < MUTATION_RATE) {
                    mutate(offspring1, domain_min, domain_max);
                    mutate(offspring2, domain_min, domain_max);
                }
                new_population.push_back(offspring1);
                if (new_population.size() < POP_SIZE) {
                    new_population.push_back(offspring2);
                }
            }
            else {
                new_population.push_back(parent1);
                if (new_population.size() < POP_SIZE) {
                    new_population.push_back(parent2);
                }
            }
        }

        population = new_population;

        // Afișează progresul
        if (generation % 100 == 0) {
            cout << "Generatia " << generation << ": Cel mai bun fitness = " << best_fitness << endl;
        }
    }

    return best_individual;
}

double calculate_mean(const vector<double>& values) {
    double sum = accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

double calculate_std_dev(const vector<double>& values, double mean) {
    double variance = 0.0;
    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    return sqrt(variance / values.size());
}

int main() {
    string function = "michalewicz"; // Schimbă între "rastrigin", "dejong", "schwefel", "michalewicz"
    double domain_min = DOMAIN_MIN;
    double domain_max = DOMAIN_MAX;

    if (function == "schwefel") {
        domain_min = -500;
        domain_max = 500;
    }
    else if (function == "michalewicz") {
        domain_min = 0;
        domain_max = M_PI;
    }

    const int RUNS = 30;
    vector<double> best_values;

    // Rulăm algoritmul genetic de 30 de ori
    cout << "Rulare Algoritm Genetic - " << RUNS << " rulari\n";
    cout << "=============================================\n";
    for (int run = 0; run < RUNS; ++run) {
        vector<double> solution = genetic_algorithm(function, domain_min, domain_max);
        double best_value = michalewicz(solution); // Schimbă funcția dacă e necesar
        best_values.push_back(best_value);

        cout << "Run " << run + 1 << ": Best Value = " << best_value << endl;
    }

    // Calculăm statistici
    double mean = calculate_mean(best_values);
    double std_dev = calculate_std_dev(best_values, mean);

    // Afișăm rezultatele în consolă
    cout << "\nRezultate Finale:\n";
    cout << "---------------------------------------------\n";
    cout << "Valori cele mai bune:\n";
    for (size_t i = 0; i < best_values.size(); ++i) {
        cout << "Rulare " << i + 1 << ": " << best_values[i] << endl;
    }

    cout << "\nMedia valorilor cele mai bune: " << mean << endl;    
    cout << "Deviația standard: " << std_dev << endl;
    cout << "=============================================\n";

    return 0;
}