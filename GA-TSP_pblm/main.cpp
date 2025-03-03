#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include <omp.h>

using namespace std;

// Structură pentru a stoca coordonatele orașelor
struct City {
    int id;
    double x, y;
};

// Matrice pentru distanțele precompute
vector<vector<double>> distanceMatrix;

// Funcție pentru calcularea distanței euclidiene între două orașe
void precomputeDistances(const vector<City>& cities) {
    int n = cities.size();
    distanceMatrix.resize(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            distanceMatrix[i][j] = sqrt(pow(cities[i].x - cities[j].x, 2) + pow(cities[i].y - cities[j].y, 2));
        }
    }
}

double getDistance(int i, int j) {
    return distanceMatrix[i][j];
}

// Funcție pentru calcularea distanței totale a unui traseu
double calculateTotalDistance(const vector<int>& route) {
    double totalDistance = 0;
    for (size_t i = 0; i < route.size() - 1; ++i) {
        totalDistance += getDistance(route[i], route[i + 1]);
    }
    totalDistance += getDistance(route.back(), route[0]);
    return totalDistance;
}

// Algoritm Genetic
class GeneticAlgorithm {
private:
    vector<City> cities;
    vector<vector<int>> population;
    int populationSize;
    int maxGenerations;
    double mutationRate;
    random_device rd;
    mt19937 gen;

public:
    GeneticAlgorithm(const vector<City>& cities, int populationSize, int maxGenerations, double mutationRate)
        : cities(cities), populationSize(populationSize), maxGenerations(maxGenerations), mutationRate(mutationRate), gen(rd()) {}

    // Generăm populația inițială
    void initializePopulation() {
        vector<int> baseRoute(cities.size());
        iota(baseRoute.begin(), baseRoute.end(), 0);

        for (int i = 0; i < populationSize; ++i) {
            shuffle(baseRoute.begin(), baseRoute.end(), gen);
            population.push_back(baseRoute);
        }
    }

vector<int> tournamentSelection() {
    int subsetSize = max(10, populationSize / 5); // 20% din populație sau minim 10 indivizi
    uniform_int_distribution<> dist(0, populationSize - 1);

    vector<int> subset(subsetSize);
    for (int i = 0; i < subsetSize; ++i) {
        subset[i] = dist(gen); // Alegem indivizi aleatoriu
    }

    // Găsim cel mai bun individ din subset
    int best = subset[0];
    for (int i = 1; i < subsetSize; ++i) {
        if (calculateTotalDistance(population[subset[i]]) < calculateTotalDistance(population[best])) {
            best = subset[i];
        }
    }

    return population[best];
}

    // Funcție de crossover (Edge Recombination)
   vector<int> crossover(const vector<int>& parent1, const vector<int>& parent2) {
    vector<int> child(cities.size(), -1);

    // Generăm două puncte de tăiere
    uniform_int_distribution<> dist(0, cities.size() - 1);
    int start = dist(gen);
    int end = dist(gen);

    if (start > end) swap(start, end);

    // Copiem segmentul din primul părinte
    for (int i = start; i <= end; ++i) {
        child[i] = parent1[i];
    }

    // Umplem restul descendentului cu gene din al doilea părinte, în ordine
    int childIndex = (end + 1) % cities.size();
    for (int i = 0; i < cities.size(); ++i) {
        int parent2Gene = parent2[(end + 1 + i) % cities.size()];

        // Dacă gena nu există deja în descendent, o adăugăm
        if (find(child.begin(), child.end(), parent2Gene) == child.end()) {
            child[childIndex] = parent2Gene;
            childIndex = (childIndex + 1) % cities.size();
        }
    }

    return child;
}


    // Funcție de mutație (reverse segment)
    void mutate(vector<int>& route) {
        uniform_real_distribution<> dist(0.0, 1.0);
        if (dist(gen) < mutationRate) {
            uniform_int_distribution<> indexDist(0, cities.size() - 1);
            int i = indexDist(gen);
            int j = indexDist(gen);
            if (i > j) swap(i, j);
            reverse(route.begin() + i, route.begin() + j + 1);
        }
    }

    // Executarea algoritmului
   // Executarea algoritmului
vector<int> run() {
    initializePopulation();

    for (int generation = 0; generation < maxGenerations; ++generation) {
        // Selectăm cei mai buni 8 indivizi
        vector<vector<int>> elites(8);
        partial_sort_copy(
            population.begin(), population.end(),
            elites.begin(), elites.end(),
            [&](const vector<int>& a, const vector<int>& b) {
                return calculateTotalDistance(a) < calculateTotalDistance(b);
            });

        vector<vector<int>> newPopulation;

        // Adăugăm elitele în noua generație
        newPopulation.insert(newPopulation.end(), elites.begin(), elites.end());

        // Generăm restul populației prin crossover și mutație
        #pragma omp parallel for
        for (int i = 8; i < populationSize; ++i) {
            vector<int> parent1 = tournamentSelection();
            vector<int> parent2 = tournamentSelection();
            vector<int> child = crossover(parent1, parent2);
            mutate(child);

            #pragma omp critical
            newPopulation.push_back(child);
        }

        population = newPopulation;

        if (generation % 100 == 0) {
            auto best = *min_element(population.begin(), population.end(), [&](const vector<int>& a, const vector<int>& b) {
                return calculateTotalDistance(a) < calculateTotalDistance(b);
            });

            cout << "Generatia " << generation << ": Cea mai buna distanta = " << calculateTotalDistance(best) << endl;
        }
    }

    // Găsim cel mai bun traseu
    return *min_element(population.begin(), population.end(), [&](const vector<int>& a, const vector<int>& b) {
        return calculateTotalDistance(a) < calculateTotalDistance(b);
    });
}

};

// Simulated Annealing
class SimulatedAnnealing {
private:
    vector<City> cities;
    double initialTemperature;
    double coolingRate;
    random_device rd;
    mt19937 gen;

public:
    SimulatedAnnealing(const vector<City>& cities, double initialTemperature, double coolingRate)
        : cities(cities), initialTemperature(initialTemperature), coolingRate(coolingRate), gen(rd()) {}

    // Generăm o soluție vecină (reverse segment)
    vector<int> getNeighbor(const vector<int>& currentRoute) {
        vector<int> neighbor = currentRoute;
        uniform_int_distribution<> dist(0, cities.size() - 1);
        int i = dist(gen);
        int j = dist(gen);
        if (i > j) swap(i, j);
        reverse(neighbor.begin() + i, neighbor.begin() + j + 1);
        return neighbor;
    }

    // Executarea algoritmului
    vector<int> run() {
        vector<int> currentRoute(cities.size());
        iota(currentRoute.begin(), currentRoute.end(), 0);
        shuffle(currentRoute.begin(), currentRoute.end(), gen);

        vector<int> bestRoute = currentRoute;
        double bestCost = calculateTotalDistance(currentRoute);

        double temperature = initialTemperature;

        while (temperature > 1e-3) {
            vector<int> neighbor = getNeighbor(currentRoute);
            double currentCost = calculateTotalDistance(currentRoute);
            double neighborCost = calculateTotalDistance(neighbor);

            if (neighborCost < currentCost || exp((currentCost - neighborCost) / temperature) > ((double)rand() / RAND_MAX)) {
                currentRoute = neighbor;
            }

            if (calculateTotalDistance(currentRoute) < bestCost) {
                bestRoute = currentRoute;
                bestCost = calculateTotalDistance(currentRoute);
            }

            temperature *= coolingRate;
        }

        return bestRoute;
    }
};

// Funcție pentru citirea instanțelor din fișiere TSPLIB
vector<City> readTSPFile(const string& filename) {
    vector<City> cities;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        if (isdigit(line[0])) {
            stringstream ss(line);
            City city;
            ss >> city.id >> city.x >> city.y;
            cities.push_back(city);
        }
    }

    return cities;
}

int main() {
    // Exemplu de utilizare
    string filename = "berlin52.tsp"; // Înlocuiți cu numele fișierului TSP
    vector<City> cities = readTSPFile(filename);
    precomputeDistances(cities);

    // Algoritm Genetic
    GeneticAlgorithm ga(cities, 200, 2000, 0.1);
    vector<int> bestRouteGA = ga.run();
    cout << "Cea mai bună solutie AG: " << calculateTotalDistance(bestRouteGA) << endl;

    // Simulated Annealing
    SimulatedAnnealing sa(cities, 1000, 0.9999);
    vector<int> bestRouteSA = sa.run();
    cout << "Cea mai bună soluie SA: " << calculateTotalDistance(bestRouteSA) << endl;

    return 0;
}
