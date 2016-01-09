#pragma once

#include "../common/Common.hpp"
#include <vector>
#include <iosfwd>


struct TrainingSample {
    vector<double> input;
    vector<double> expectedOutput;

    TrainingSample(const vector<double> &input, const vector<double> expectedOutput) :
      input(input), expectedOutput(expectedOutput) {}
};

std::ostream& operator<<(std::ostream& stream, const TrainingSample& ts);

class Network {
public:
  Network(const vector<unsigned> &layerSizes);
  virtual ~Network();

  vector<double> Process(const vector<double> &input);
  void Train(const vector<TrainingSample> &samples, double learnRate);

private:
  struct NetworkImpl;
  uptr<NetworkImpl> impl;
};
