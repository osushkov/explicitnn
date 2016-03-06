
#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>

#include "util/Util.hpp"
#include "neuralnetwork/Network.hpp"

using namespace std;

vector<TrainingSample> getTrainingData(unsigned howMany) {
  vector<TrainingSample> trainingData;

  float centreX = 0.75f, centreY = 0.6f;
  float radius = 0.4f;
  for (unsigned i = 0; i < howMany; i++) {
    float x = Util::RandInterval(-1.0, 1.0);
    float y = Util::RandInterval(-1.0, 1.0);

    float d = sqrt((x - centreX) * (x - centreX) + (y - centreY) * (y - centreY));
    float r = d < radius ? 1.0f : 0.0f;

    trainingData.push_back(TrainingSample{{x, y}, {r}});
  }

  // trainingData.push_back(TrainingSample{{1.0, 0.0}, {1.0}});
  // trainingData.push_back(TrainingSample{{0.0, 1.0}, {1.0}});
  // trainingData.push_back(TrainingSample{{1.0, 1.0}, {0.0}});
  // trainingData.push_back(TrainingSample{{0.0, 0.0}, {0.0}});
  return trainingData;
}


void trainNetwork(Network &network, const std::vector<TrainingSample> &trainingSamples, unsigned iterations) {
  float startLearningRate = 0.2;
  float endLearningRate = 0.01;

  for (unsigned i = 0; i < iterations; i++) {
    float lr = startLearningRate + (endLearningRate - startLearningRate) * i / (float)iterations;
    network.Train(trainingSamples, lr);
  }
}

void evaluateNetwork(Network &network, const std::vector<TrainingSample> &evalSamples) {
  for (const auto& es : evalSamples) {
    auto result = network.Process(es.input);
    cout << es << " -> ";
    for (auto v : result) {
      cout << v << " ";
    }
    cout << endl;
  }
}

int main() {
  srand(1234);

  Network network({2, 3, 1});

  vector<TrainingSample> trainingSamples = getTrainingData(500);
  trainNetwork(network, trainingSamples, 1000);

  vector<TrainingSample> evalSamples = getTrainingData(200);
  evaluateNetwork(network, evalSamples);

  return 0;
}
