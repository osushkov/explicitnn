
#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>

#include "util/Util.hpp"
#include "neuralnetwork/Network.hpp"

using namespace std;

vector<TrainingSample> getTrainingData(void) {
  vector<TrainingSample> trainingData;

  double centreX = 0.75, centreY = 0.6;
  double radius = 0.5;
  for (unsigned i = 0; i < 20; i++) {
    double x = Util::RandInterval(-1.0, 1.0);
    double y = Util::RandInterval(-1.0, 1.0);

    double d = sqrt((x - centreX) * (x - centreX) + (y - centreY) * (y - centreY));
    double r = d < radius ? 1.0 : 0.0;

    trainingData.push_back(TrainingSample{{x, y}, {r}});
  }

  // trainingData.push_back(TrainingSample{{1.0, 0.0}, {1.0}});
  // trainingData.push_back(TrainingSample{{0.0, 1.0}, {1.0}});
  // trainingData.push_back(TrainingSample{{1.0, 1.0}, {0.0}});
  // trainingData.push_back(TrainingSample{{0.0, 0.0}, {0.0}});
  return trainingData;
}


void trainNetwork(Network &network, const std::vector<TrainingSample> &trainingSamples, unsigned iterations) {
  double startLearningRate = 0.2;
  double endLearningRate = 0.01;

  for (unsigned i = 0; i < iterations; i++) {
    double lr = startLearningRate + (endLearningRate - startLearningRate) * i / (double)iterations;
    network.Train(trainingSamples, lr);
  }
}

void evaluateNetwork(Network &network, const std::vector<TrainingSample> &evalSamples) {
  for (const auto& es : evalSamples) {
    auto result = network.Process(es.input);
    cout << es << " -> ";
    for (auto v : result) {
      cout << (v > 0.5) << " ";
    }
    cout << endl;
  }
}

int main() {
  Network network({2, 3, 1});
  vector<TrainingSample> td = getTrainingData();
  trainNetwork(network, td, 10000);
  evaluateNetwork(network, td);
  return 0;
}
